import os
import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, CLIPModel, BertModel, ResNetModel, \
    ChineseCLIPVisionModel
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(dim=0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x)


class BaseFeatureEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.base_model = ChineseCLIPModel.from_pretrained(args.base_path)
        self.base_model.logit_scale.requires_grad = False
        for param in self.base_model.vision_model.parameters():
            param.requires_grad = False
        for param in self.base_model.visual_projection.parameters():
            param.requires_grad = False

        self.text_ln = nn.LayerNorm(512)
        self.visual_ln = nn.LayerNorm(512)

    def forward(self, input_ids, visual_feats):
        # shape of input_ids: B * num_video * num_frames * len_sent
        # shape of attention_mask: B * num_video * num_frames * len_sent
        # shape of pixel_values: B * num_videos * num_frames * 3 * 224 * 224

        B, num_videos, num_frames, len_sent = input_ids.shape
        input_ids = input_ids.reshape(-1, len_sent)
        attention_mask = (input_ids != 0).bool()
        attention_mask = attention_mask.reshape(-1, len_sent)

        text_outputs = self.base_model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask).reshape(B, num_videos, num_frames, -1)

        text_feats = self.text_ln(text_outputs)

        visual_feats = self.visual_ln(visual_feats)

        return text_feats, visual_feats


class MMVideoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.pe = PositionalEncoding(d_model=args.d_model)
        self.prefix = nn.Parameter(torch.randn([args.d_model]), requires_grad=True)

        decoder_layer = nn.TransformerDecoderLayer(d_model=args.d_model, nhead=args.nhead,
                                                   dim_feedforward=args.dim_feedforward,
                                                   batch_first=True,
                                                   dropout=args.dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.num_en_layers)

        self.final_ln = nn.LayerNorm(args.d_model)

    def forward(self, text_feats, visual_feats):
        # shape of src: B * num_video * num_chunks * word_length
        # shape of frame_mask: B * num_video * num_frames
        # shape of word_mask: B * num_video * num_frames * len_sent

        B, num_videos, num_frames, vdim = visual_feats.shape
        _, _, num_chunks, tdim = text_feats.shape

        text_feats = text_feats.reshape(B * num_videos, num_chunks, -1)

        text_feats = torch.cat([self.prefix.reshape(1, 1, -1).expand(B * num_videos, -1, -1), text_feats], dim=1)

        visual_feats = visual_feats.reshape(B * num_videos, -1, visual_feats.shape[-1])

        # positional embedding
        text_feats = self.pe(text_feats)
        visual_feats = self.pe(visual_feats)

        feats = self.decoder(text_feats, visual_feats)[:, 0, :].reshape(B, num_videos, -1)

        feats = self.final_ln(feats)

        return feats


class CollectionEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=args.d_model, nhead=args.nhead,
                                                   dim_feedforward=args.dim_feedforward,
                                                   batch_first=True,
                                                   dropout=args.dropout)
        self.coll_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_en_layers)

    def forward(self, src, mask=None):
        # shape of input_ids: B * num_videos * feats_dim
        # shape of mask: B * num_videos

        outputs = self.coll_encoder(src, src_key_padding_mask=~mask.bool())

        return outputs


class OrderDecoder(nn.Module):
    def __init__(self, args, emb):
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(d_model=args.d_model,
                                                   nhead=args.nhead,
                                                   dim_feedforward=args.dim_feedforward,
                                                   batch_first=True,
                                                   dropout=args.dropout)

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.num_de_layers)

        self.max_len = args.max_videos

        self.emb = emb

        self.generator = nn.Linear(args.d_model, self.max_len)

        self.pos_emb = nn.Embedding(self.max_len + 2, args.d_model)

        self.beam_size = args.beam_size

    def forward(self, en_outputs, gt_orders, mask):
        # shape of feats: B * num_videos * d_model
        # shape of mask: B * num_videos
        # shape of gt_orders: B * num_videos [[3, 4,...,1, -1, -1],...]

        B, num_videos, d_model = en_outputs.shape
        # assert num_videos == self.max_len

        mem_key_padding_mask = ~mask.bool()
        tgt_mask = self.generate_square_subsequent_mask(mask.shape[1], mask.device)
        tgt_mask = tgt_mask.to(en_outputs.dtype)
        tgt_key_padding_mask = ~torch.cat([torch.ones([B, 1], device=mask.device), mask], dim=1).bool()
        tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]
        order = gt_orders.clone()
        order[order == -1] = 1000
        sorted_ids = torch.argsort(order, dim=1)
        sorted_ids[order == 1000] = self.max_len + 1  # pad_token
        sorted_ids = torch.cat([(torch.ones([B, 1], device=mask.device) * self.max_len).long(), sorted_ids],
                               dim=1)  # add start_token

        de_inputs = self.emb(sorted_ids)  # shape: B * (max_len + 1) * d

        pos = torch.arange(num_videos + 1, device=mask.device).unsqueeze(0).expand(B, -1)
        de_inputs = de_inputs + self.pos_emb(pos)

        # add order embedding
        en_emb_ids = torch.arange(num_videos, device=mask.device).unsqueeze(0).expand(B, -1)
        en_emb_ids = en_emb_ids.clone()
        en_emb_ids[mask == 0] = num_videos + 1
        en_outputs1 = en_outputs + self.emb(en_emb_ids)

        outputs = self.decoder(memory=en_outputs1,
                               tgt=de_inputs[:, :-1, :],
                               tgt_mask=tgt_mask,
                               memory_key_padding_mask=mem_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)

        preds = self.generator(outputs)  # output shape: B * num_videos * 48

        return preds

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def beam_search_decode(self, en_outputs, mask, ACL):
        B, num_videos, d_model = en_outputs.shape
        beam_width = self.beam_size

        final_preds = []
        start_token = self.max_len

        for i in range(B):
            beams = [Beam(seq=[start_token], score=0)]
            gt_lens = mask[i].sum()
            if gt_lens < beam_width:
                beam_width = gt_lens
            en_outs = en_outputs[i][:gt_lens, :]

            en_emb_ids = torch.arange(gt_lens, device=mask.device)
            en_emb_ids = en_emb_ids.clone()
            en_outs1 = en_outs + self.emb(en_emb_ids)

            for j in range(gt_lens):
                new_beams = []
                for beam in beams:
                    seq = torch.tensor(beam.seq, device=mask.device).long()

                    de_inputs = self.emb(seq)
                    pos = torch.arange(len(seq), device=mask.device)
                    de_inputs = de_inputs + self.pos_emb(pos)

                    tgt_mask = self.generate_square_subsequent_mask(de_inputs.shape[0], mask.device)
                    tgt_mask = tgt_mask.to(en_outputs.dtype)
                    outputs = self.decoder(memory=en_outs1,
                                           tgt=de_inputs,
                                           tgt_mask=tgt_mask)
                    logits = self.generator(outputs[-1, :])  # shape: max_len
                    preds = F.softmax(logits[: gt_lens], dim=-1)  # shape: gt_lens

                    if j > 0:
                        start = en_outs[beam.seq[-1]]
                        start_end = torch.cat([start.unsqueeze(0).expand(gt_lens, -1), en_outs], dim=-1)
                        next_scores = ACL.coh_func(start_end).reshape(-1) / ACL.tau
                        next_scores = F.softmax(next_scores, dim=0)
                        preds += next_scores

                    beam_mask = torch.ones([gt_lens], device=mask.device)
                    beam_mask[torch.LongTensor(beam.seq[1:])] = 0
                    preds = beam_mask * preds

                    topk_probs, topk_indices = torch.topk(preds, k=beam_width, dim=-1)
                    for k in range(beam_width):
                        new_beam = Beam(seq=beam.seq + [topk_indices[k].item()],
                                        score=beam.score + topk_probs[k].item())
                        new_beams.append(new_beam)

                    # Sort the new beams by their score
                    new_beams.sort(key=lambda b: b.score, reverse=True)
                    # Prune the beams
                    beams = new_beams[:beam_width]

            final_preds.append(beams[0].seq[1:])

        return final_preds


class Beam:
    def __init__(self, seq, score):
        self.seq = seq
        self.score = score


class SucPred(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.d_model = args.d_model
        self.coh_func = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model),
                                      nn.LeakyReLU(0.1),
                                      nn.Linear(self.d_model, 1))

        self.tau = args.tau

    def forward(self, inputs, mask, order):
        # shape of inputs: (batch_size, seq_len, d_model)
        # shape of mask: (batch_size, seq_len)
        # shape of order: (batch_size, seq_len)

        losses = 0
        for x, m, o in zip(inputs, mask, order):
            length = m.sum().item()
            x = x[:length]
            o = o[:length]
            idx = torch.tensor(list(itertools.product(np.arange(length), repeat=2)), device=x.device)
            feats = torch.cat([x[idx[:, 0]], x[idx[:, 1]]], dim=-1)  # shape: seq_len^2 * 2d_model
            coh = self.coh_func(feats)
            gt = (o[idx[:, 0]] - o[idx[:, 1]] == -1).int()

            coh1 = coh.reshape(length, -1) / self.tau
            gt1 = gt.reshape(length, -1).argmax(1)
            last = torch.argmax(o)
            gt1[last] = -1
            loss = F.cross_entropy(coh1, gt1, ignore_index=-1)
            losses += loss

        return losses / inputs.shape[0]

