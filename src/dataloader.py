import numpy as np
import torch
import random
import os
import json
from torch.utils.data import Dataset, DataLoader

from transformers import ChineseCLIPProcessor


class VODataset(Dataset):
    def __init__(self, base_path, data_root, split='train', max_videos=10):
        self.split = split
        self.split_path = os.path.join(data_root, f'{split}.txt')
        self.max_videos = max_videos  # The max length of the collection we aim to handle.

        with open(self.split_path, 'r', encoding='utf-8') as f:
            coll_ids = f.readlines()
        self.coll_ids = [id_.replace('\n', '') for id_ in coll_ids]

        self.subtitles_path = os.path.join(data_root, 'subtitles')
        self.visual_feats_path = os.path.join(data_root, 'clip_visual_32')

        self.max_words = 64  # max words of a chunk.
        self.max_chunks = 16  # max chunks of a doc.
        self.max_frames = 16  # max frames of a video.

        self.processor = ChineseCLIPProcessor.from_pretrained(base_path)

    def __len__(self):
        return len(self.coll_ids)

    def __getitem__(self, item):
        coll_id = self.coll_ids[item]
        coll_subt_path = os.path.join(self.subtitles_path, str(coll_id) + '.json')

        with open(coll_subt_path, 'r') as f:
            jsondata = f.read()
        coll_subts = json.loads(jsondata)  # Dict
        docids = list(coll_subts.keys())

        if len(docids) > self.max_videos:
            docids = docids[:self.max_videos]

        gt_order = list(range(len(docids)))

        random.shuffle(gt_order)
        shuffled_docids = [docids[idx] for idx in gt_order]

        shuffled_ids = []
        shuffled_vfeats = []

        for docid in shuffled_docids:

            frame_subts = coll_subts[docid][1:]
            if not frame_subts:
                frame_subts = ['无']
            # subts = frame_subts
            subts = '。'.join(frame_subts)
            subts = subts + '。'
            subts = [subts[i:i + self.max_words] for i in range(0, len(subts), self.max_words)]
            if len(subts) > self.max_chunks:
                subts = subts[0:int(self.max_chunks/2)] + subts[-int(self.max_chunks/2):]
            subts = [coll_subts[docid][0]] + subts

            input_ids = self.processor(text=subts, padding='max_length', max_length=self.max_words, return_tensors='pt',
                                       truncation=True)['input_ids']
            shuffled_ids.append(input_ids)

            visual_path = os.path.join(self.visual_feats_path, str(coll_id), docid + '.npy')

            if os.path.isfile(visual_path):
                feature = torch.tensor(np.load(visual_path))
            else:
                feature = torch.zeros([self.max_frames, 512]).float()
                # print('No such file: ', visual_path)
            if len(feature) > self.max_frames:
                ids = torch.linspace(0, len(feature) - 1, self.max_frames).long()
                feature = torch.index_select(feature, 0, ids)
            else:
                feature = torch.cat([feature, torch.zeros([self.max_frames - len(feature), 512])], dim=0)

            shuffled_vfeats.append(feature)
        shuffled_vfeats = torch.stack(shuffled_vfeats, dim=0)

        return shuffled_ids, shuffled_vfeats, gt_order


def collate_fn(batch):
    bs = len(batch)
    max_num_videos = max(len(batch[i][0]) for i in range(bs))
    max_num_chunks = max(max(len(batch[i][0][j]) for j in range(len(batch[i][0]))) for i in range(bs))

    input_ids = torch.zeros([bs, max_num_videos, max_num_chunks, batch[0][0][0].shape[-1]]).long()
    video_feats = torch.zeros([bs, max_num_videos, *batch[0][1][0].shape]).float()
    gt_orders = torch.ones([bs, max_num_videos]) * (-1)

    for i in range(bs):
        shuffled_ids, shuffled_vfeats, gt_order = batch[i]
        for j in range(len(shuffled_ids)):
            input_ids[i][j][:len(shuffled_ids[j])] = shuffled_ids[j]
        video_feats[i][:len(shuffled_vfeats)] = shuffled_vfeats
        gt_orders[i][:len(gt_order)] = torch.tensor(gt_order)

    return input_ids, video_feats, gt_orders.long()


def truncate_sentences(sentences, max_words):
    sent_lens = np.array([len(sent) for sent in sentences])
    while sent_lens.sum() > max_words and len(sentences) > 1:
        sentences.remove(sentences[int(len(sentences) / 2)])
        sent_lens = np.array([len(sent) for sent in sentences])
    return sentences


def get_loaders(args):
    train_set = VODataset(args.base_path, args.data_root, split='train', max_videos=args.max_videos)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              num_workers=4,
                              collate_fn=collate_fn,
                              pin_memory=True,
                              drop_last=True)

    test_set = VODataset(args.base_path, args.data_root, split='test', max_videos=args.max_videos)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=1,
                             num_workers=4,
                             collate_fn=collate_fn)

    val_set = VODataset(args.base_path, args.data_root, split='val', max_videos=args.max_videos)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=collate_fn)

    return train_loader, test_loader, val_loader
