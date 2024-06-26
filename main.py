import random

import torch.backends.cudnn as cudnn
from torch.testing._internal.common_quantization import AverageMeter

from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
import torch.distributed as dist
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
# from torchsummary import summary
from utils import count_params, meanIOU, color_map


import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("here are the all the devices: ", device)
# torch.cuda.empty_cache()
torch.set_warn_always(True)
MODE = None

def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes'], default='pascal')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv3plus')

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)

    parser.add_argument('--save-path', type=str, required=True)

    # arguments for ST++
    parser.add_argument('--reliable-id-path', type=str)
    parser.add_argument('--plus', dest='plus', default=False, action='store_true',
                        help='whether to use ST++')

    parser.add_argument("--first", type=int, default=2)
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument('--seed', default=114154, type=int)

    args = parser.parse_args()
    return args


def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.plus and args.reliable_id_path is None:
        exit('Please specify reliable-id-path in ST++.')

    torch.distributed.init_process_group(backend='nccl')

    import random
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)




    model = DeepLabV3Plus(args.backbone, 21)

    head_lr_multiple = 10.0
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                     {'params': [param for name, param in model.named_parameters()
                                 if 'backbone' not in name],
                      'lr': args.lr * head_lr_multiple}],
                    lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = CrossEntropyLoss(ignore_index=255).cuda(args.local_rank)


    if args.local_rank == 0:
        print('\nParams: %.1fM' % count_params(model))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    device = torch.device('cuda', args.local_rank)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=False)


    global MODE
    MODE = 'train'

    # labelled dataset
    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.labeled_id_path)
    trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
    train_sampler = DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             pin_memory=True, num_workers=1, drop_last=True, sampler=train_sampler)

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    val_sampler = DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1,
                            pin_memory=True, num_workers=1, drop_last=False, sampler=val_sampler)

    # <====================== Supervised training with labeled images (SupOnly) ======================>
    if  args.local_rank == 0:
        print('\n================> Total stage 1/%i: '
            'Supervised training on labeled images (SupOnly)' % (6 if args.plus else 3))

    best_model, checkpoints = train(model, trainloader, valloader, criterion, optimizer, args)


    """
        ST framework without selective re-training
    """
    if not args.plus:
        # <============================= Pseudo label all unlabeled images =============================>
        print('\n\n\n================> Total stage 2/3: Pseudo labeling all unlabeled images')

        dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
        dataset_sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=4, drop_last=False, sampler=dataset_sampler)

        label(best_model, dataloader, args)


        # <======================== Re-training on labeled and unlabeled images ========================>
        print('\n\n\n================> Total stage 3/3: Re-training on labeled and unlabeled images')

        MODE = 'semi_train'

        trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                               args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
        trainset_sampler = DistributedSampler(trainset)
        trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                 pin_memory=True, num_workers=16, drop_last=True, sampler=trainset_sampler)

        model = DeepLabV3Plus(args.backbone, 21)
        head_lr_multiple = 10.0
        optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                         {'params': [param for name, param in model.named_parameters()
                                     if 'backbone' not in name],
                          'lr': args.lr * head_lr_multiple}],
                        lr=args.lr, momentum=0.9, weight_decay=1e-4)
        criterion = CrossEntropyLoss(ignore_index=255).cuda(args.local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        device = torch.device('cuda', args.local_rank)
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parameters=False)

        train(model, trainloader, valloader, criterion, optimizer, args)


        return



def train(model, trainloader, valloader, criterion, optimizer, args):
    iters = 0
    total_iters = len(trainloader) * args.epochs

    previous_best = 0.0

    global MODE
    device = torch.device('cuda', args.local_rank)
    if MODE == 'train':
        checkpoints = []

    for epoch in range(args.epochs):
        if args.local_rank == 0:
            print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
                (epoch, optimizer.param_groups[0]["lr"], previous_best))
        total_loss = 0
        trainloader.sampler.set_epoch(epoch)

        model.train()
        tbar = tqdm(trainloader)

        # input image shape is torch.Size([16, 3, 321, 321])
        for i, (img, mask) in enumerate(tbar):
            img, mask = img.to(device), mask.to(device)
            # print(summary(model, (3, 321, 321), 16))
            pred = model(img)
            loss = criterion(pred, mask)
            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

            if args.local_rank == 0:
                tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

        metric = meanIOU(num_classes=21)

        model.eval()
        tbar = tqdm(valloader)

        with torch.no_grad():
            for img, mask, _ in tbar:
                img = img.cuda()
                pred = model(img)
                pred = torch.argmax(pred, dim=1)

                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]
                if args.local_rank == 0:
                    tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

        mIOU *= 100.0
        if mIOU > previous_best:
            if previous_best != 0:
                file_path = os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best))
                if os.path.exists(file_path):
                    os.remove(file_path)

            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))

            best_model = deepcopy(model)

        if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            checkpoints.append(deepcopy(model))

    if MODE == 'train':
        return best_model, checkpoints

    return best_model



def label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)

    metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
    cmap = color_map(args.dataset)
    device = torch.device('cuda', args.local_rank)
    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.to(device)
            pred = model(img)
            # pred = model(img)

            pred = torch.argmax(pred, dim=1).cpu()

            metric.add_batch(pred.numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1]

            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)

            pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))

            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))



# 这下面的是用来控制第一个gpu上跑多少个batch，剩下的gpu上跑多少个batch的
import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel._functions import Scatter


def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except Exception:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class BalancedDataParallel(DataParallel):

    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids

        if inputs[0].size()[0] == 1:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids)
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]
        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)

def init_basic_elems(args):

    model = DeepLabV3Plus(args.backbone, 21)
    head_lr_multiple = 10.0
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                     {'params': [param for name, param in model.named_parameters()
                                 if 'backbone' not in name],
                      'lr': args.lr * head_lr_multiple}],
                    lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # model = DataParallel(model)
    # model = BalancedDataParallel(args.first, model, dim=0)

    return model, optimizer
if __name__ == '__main__':
    args = parse_args()

    if args.epochs is None:
        args.epochs = {'pascal': 80, 'cityscapes': 240}[args.dataset]
    if args.lr is None:
        args.lr = {'pascal': 0.001, 'cityscapes': 0.004}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'pascal': 321, 'cityscapes': 721}[args.dataset]
    if args.local_rank == 0:
        print()
        print(args)

    main(args)
