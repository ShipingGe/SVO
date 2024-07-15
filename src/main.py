
import lightning.pytorch as pl

from modeling import *

from dataloader import get_loaders
from argparse import ArgumentParser
from evaluation import Kendall_Tau, Accuracy, PMR
from torch.distributed.fsdp.wrap import wrap

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


class ReorderModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.emb = nn.Embedding(args.max_videos + 2, args.d_model)  # +2: start_token, pad_token
        # self.emb.weight.requires_grad = False

        self.frame_encoder = BaseFeatureEncoder(args)
        self.video_encoder = MMVideoEncoder(args)
        self.coll_encoder = CollectionEncoder(args)
        self.SP = SucPred(args)
        self.order_decoder = OrderDecoder(args, self.emb)

        # evaluation metrics
        self.accuracy = Accuracy()
        self.kendall_tau = Kendall_Tau()
        self.pmr = PMR()

        self.short_acc = Accuracy(type_='s')
        self.short_tau = Kendall_Tau(type_='s')
        self.short_pmr = PMR(type_='s')

        self.long_acc = Accuracy(type_='l')
        self.long_tau = Kendall_Tau(type_='l')
        self.long_pmr = PMR(type_='l')

    def training_step(self, batch, batch_idx):

        input_ids, video_inputs, gt_orders = batch
        coll_mask = (gt_orders != -1).bool()
        text_feats, visual_feats = self.frame_encoder(input_ids, video_inputs)
        video_feats = self.video_encoder(text_feats, visual_feats)

        outputs = self.coll_encoder(video_feats, coll_mask)
        acl_loss = self.SP(outputs, coll_mask, gt_orders)
        order_logits = self.order_decoder(outputs, gt_orders, coll_mask)

        # generate the gt_ids from the gt_orders
        order = gt_orders.clone()
        order[order == -1] = 1000  # a value that is larger than any collections.
        gt_ids = torch.argsort(order, dim=1)
        gt_ids[order == 1000] = -1
        de_ce_loss = F.cross_entropy(order_logits.reshape(-1, order_logits.shape[-1]),
                                     gt_ids.reshape(-1),
                                     ignore_index=-1)

        loss = de_ce_loss + acl_loss

        self.log('loss', loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):

        input_ids, video_inputs, gt_orders = batch
        coll_mask = (gt_orders != -1).bool()
        text_feats, visual_feats = self.frame_encoder(input_ids, video_inputs)
        video_feats = self.video_encoder(text_feats, visual_feats)

        outputs = self.coll_encoder(video_feats, coll_mask)

        pred_orders = self.order_decoder.beam_search_decode(outputs, coll_mask, self.SP)
        # print(pred_orders)

        self.accuracy(pred_orders, gt_orders, coll_mask)
        self.kendall_tau(pred_orders, gt_orders, coll_mask)
        self.pmr(pred_orders, gt_orders, coll_mask)

        self.short_acc(pred_orders, gt_orders, coll_mask)
        self.short_tau(pred_orders, gt_orders, coll_mask)
        self.short_pmr(pred_orders, gt_orders, coll_mask)

        self.long_acc(pred_orders, gt_orders, coll_mask)
        self.long_tau(pred_orders, gt_orders, coll_mask)
        self.long_pmr(pred_orders, gt_orders, coll_mask)

    def on_test_epoch_end(self) -> None:
        self.log("Accuracy", self.accuracy)
        self.log("Kenall_tau", self.kendall_tau)
        self.log("PMR", self.pmr)

        self.log("Short Accuracy", self.short_acc)
        self.log("Short Tau", self.short_tau)
        self.log("Short PMR", self.short_pmr)

        self.log("Long Accuracy", self.long_acc)
        self.log("Long Tau", self.long_tau)
        self.log("Long PMR", self.long_pmr)

        self.log("Num of test samples", self.kendall_tau.total)

    def validation_step(self, batch, batch_idx):

        input_ids, video_inputs, gt_orders = batch
        coll_mask = (gt_orders != -1).bool()
        text_feats, visual_feats = self.frame_encoder(input_ids, video_inputs)
        video_feats = self.video_encoder(text_feats, visual_feats)

        outputs = self.coll_encoder(video_feats, coll_mask)

        pred_orders = self.order_decoder.beam_search_decode(outputs, coll_mask, self.SP)

        self.val_acc(pred_orders, gt_orders, coll_mask)

    def on_validation_epoch_end(self) -> None:
        self.log('Val_Acc', self.val_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                      lr=args.learning_rate,
                                      betas=(0.9, 0.95),
                                      eps=1e-05,
                                      weight_decay=0.01)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Number of training batch size.")

    parser.add_argument("--base_path", type=str,
                        default="Path to models--TencentARC--QA-CLIP-ViT-B-16",
                        help="Path to the pretrained base model.")
    parser.add_argument("--data_root", type=str,
                        default='../dataset',
                        help='Path to the dataset.')
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--num_en_layers", type=int, default=2, help="Number of the layers of the encoder.")
    parser.add_argument("--num_de_layers", type=int, default=2, help="Number of the layers of the decoder.")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--eval", action='store_true', default=False, help='Whether to evaluate the model only.')
    parser.add_argument("--ckpt_path", type=str,
                        default='',
                        help='Path to the saved checkpoint.')

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--accelerator", type=str, default='gpu')

    parser.add_argument("--max_videos", type=int, default=16, help="The maximum number of videos per collection.")

    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--beam_size", type=int, default=4)

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    train_loader, test_loader, val_loader = get_loaders(args)

    model = ReorderModel(args)

    if args.eval:
        trainer = pl.Trainer(devices=1, num_nodes=1,
                             inference_mode=True,
                             accelerator=args.accelerator,
                             default_root_dir='./')
        trainer.test(model, test_loader, ckpt_path=args.ckpt_path)

    else:
        trainer = pl.Trainer(max_epochs=args.epochs, accelerator=args.accelerator, precision='16-mixed',
                             default_root_dir='./', gradient_clip_val=1.0,
                             strategy='ddp_find_unused_parameters_true')
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        torch.distributed.destroy_process_group()
        if trainer.global_rank == 0:
            trainer2 = pl.Trainer(devices=1, num_nodes=1, inference_mode=True,
                                  accelerator=args.accelerator,
                                  default_root_dir='./')
            trainer2.test(model, test_loader)
