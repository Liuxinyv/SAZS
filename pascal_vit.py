#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import argparse
import json
import logging
import math
import os
from scipy.ndimage.morphology import distance_transform_edt
import pdb
from os.path import exists, join, split
import threading
from torchvision.transforms import Resize
import time
from pathlib import Path
import numpy as np
import shutil
import sys
import PIL.Image as Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from CLIP.clip import clip
from fewshot_data.common import utils
import drn
import data_transforms as transforms
from model.models import DPTSegmentationModel
from utils import JointEdgeSegLoss
try:
    from modules import batchnormsync
except ImportError:
    pass
from torch.utils.tensorboard import SummaryWriter
from fewshot_data.common.vis import Visualizer
import torch.nn.functional as F
from fewshot_data.data.dataset import FSSDataset
CITYSCAPE_PALETTE = np.asarray([
    [0, 0, 0],
    [255,128,255],
    [255,255,128],
    [128, 64, 128],#dark purple
    [244, 35, 232],#shellow purple
    [70, 70, 70],#gray
    [102, 102, 156],#gray blue
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 100, 100]], dtype=np.uint8)


def validate(args,val_loader,val_dataset, model, criterion, eval_score=None, print_freq=1,texts=None,lmodel=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()
    model.eval()
    averagemeter = AverageMeter2(args,val_dataset)

    writer = SummaryWriter(log_dir='runs/pascal_vit_edge')

    end = time.time()
    for i, (input, target,edge_gts,class_sample,name) in enumerate(val_loader):
        target, ignore = extract_ignore_idx(target, class_sample)
        h, w = input.size()[2:4]
        with torch.no_grad():
            torch_resize = Resize([512, 512])
            input = torch_resize(input)
            input = input.cuda()
            target = target.cuda()
            ignore = ignore.cuda()
            input_var = torch.autograd.Variable(input, volatile=True)
            features = model(input_var)
            text_features = [lmodel.encode_text(texts[class_i]).detach() for class_i in class_sample]
            text_features = torch.cat([pred for pred in text_features]).reshape(int(target.size(0)), 2, 512)
            text_features = text_features.to(torch.float32)
            result = torch.einsum('abcd,aeb->aecd', (features[0], text_features))
            tempsoftmax = nn.LogSoftmax()
            result = tempsoftmax(result)

            area_inter, area_union,beta,beta_f = fast_hist(result.argmax(axis=1), target, ignore)
            averagemeter.update(area_inter, area_union, class_sample.cuda())
            miou, mious, fb_iou = averagemeter.compute_iou()
            logger.info('===> mIoU {mIoU:.3f}'.format(
                mIoU=miou))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'miou {miou:.3f}\t'
                        'fb_iou {fb_iou:.4f}\t'
                        'Score {score.val:.3f} ({score.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses, miou=miou, fb_iou=fb_iou, score=score))
    val_miou, val_mious, val_fb_iou = averagemeter.compute_iou()
    logger.info(' '.join('{:.03f}'.format(i) for i in val_mious))
    logger.info('===> mIoU {mIoU:.3f}'.format(mIoU=val_miou))
    return val_miou, val_fb_iou


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeter2(object):
    """Computes and stores the average and current value"""

    def __init__(self,args,val_dataset):
        self.benchmark = args.benchmark
        self.class_ids_interest = torch.tensor(val_dataset.class_ids).cuda()
        if self.benchmark == 'pascal':
            self.nclass = 20
        elif self.benchmark == 'coco':
            self.nclass = 80
        self.intersection_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.union_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.ones = torch.ones_like(self.union_buf)
        self.loss_buf = []

    def update(self,inter_b, union_b, class_id):
        self.intersection_buf.index_add_(1, class_id, inter_b.float())
        self.union_buf.index_add_(1, class_id, union_b.float())

    def compute_iou(self):
        iou = self.intersection_buf.float() / \
              torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100
        mious = iou[1] * 100

        fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
                  self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

        return miou, mious,fb_iou


def accuracy(output, target,ignore):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    if ignore is not None:
        assert torch.logical_and(ignore, target).sum() == 0
        ignore *= 255
        target = target + ignore
        pred[target == 255] = 255
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255]
    correct = correct.view(-1)
    try:
        score = correct.float().sum(0).mul(100.0 / correct.size(0))
        return score.item()
    except:
        return 0

def extract_ignore_idx(mask, class_id):
    boundary = (mask / 255).floor()
    mask[mask != class_id + 1] = 0
    mask[mask == class_id + 1] = 1

    return mask, boundary


def train(args,train_loader, model,criterion, optimizer, epoch,
          eval_score=None, print_freq=1,texts=None,lmodel=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    writer = SummaryWriter(log_dir='runs/pascal_vit_edge_f'+str(args.fold))
    end = time.time()

    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = batch['query_img']
        target = batch['query_mask']
        class_sample = batch['class_id']
        name = batch['query_name']
        edge_gts = batch['edge_gts']
        small_target = torch.zeros(int(target.size(0)), int(target.size(1)/8), int(target.size(2)/8))
        small_edge_gts = torch.zeros(int(edge_gts.size(0)), int(edge_gts.size(1)/8), int(edge_gts.size(2)/8))
        small_ignore = small_target.clone()
        for index in range(0, target.size(0)):
            temp = target[index, :, :]
            temp = cv2.resize(temp.numpy(), (int(target.size(1) / 8), int(target.size(2) / 8)),
                              interpolation=cv2.INTER_NEAREST)
            temp, ignore = extract_ignore_idx(torch.tensor(temp), class_sample[index])
            small_target[index, :, :] = temp
            small_ignore[index, :, :] = ignore

            temp_e = edge_gts[index, :, :]
            temp_e = cv2.resize(temp_e.numpy(), (int(edge_gts.size(1) / 8), int(edge_gts.size(2) / 8)),
                                interpolation=cv2.INTER_NEAREST)
            small_edge_gts[index, :, :] = torch.tensor(temp_e)
        target = small_target
        target = target.long()
        ignore = small_ignore
        ignore=ignore.long()
        edge_gts = small_edge_gts
        edge_gts = torch.unsqueeze(edge_gts, 1)
        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()

        input = input.cuda()
        target = target.cuda()
        ignore=ignore.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        ignore_var = torch.autograd.Variable(ignore)

        edge_out,features = model(input_var)
        text_features = [lmodel.encode_text(texts[class_i]).detach() for class_i in class_sample]
        text_features = torch.cat([pred for pred in text_features]).reshape(int(target.size(0)),2,512)
        text_features = text_features.to(torch.float32)
        result = torch.einsum('abcd,aeb->aecd', (features, text_features))
        tempsoftmax = nn.LogSoftmax()
        result = tempsoftmax(result)
        loss = criterion((result, edge_out), (target_var, edge_gts))
        losses.update(loss.item(), input.size(0))
        if eval_score is not None:
            scores.update(eval_score(result, target_var,ignore_var), input.size(0))
        writer.add_scalar("loss", losses.val, epoch * len(train_loader) + i)
        writer.add_scalar("avg_loss", losses.avg, epoch * len(train_loader) + i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=scores))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar',new_dir=None):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,new_dir+'/model_best.pth.tar')


def train_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size
    fold = args.fold

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DPTSegmentationModel(2, backbone=args.arch)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        for name, param in checkpoint['state_dict'].items():
            name = name[7:]
            single_model.state_dict()[name].copy_(param)
    # model = torch.nn.DataParallel(single_model).cuda()
    model = single_model.cuda()
    criterion = JointEdgeSegLoss(
        ignore_index=255).cuda()
    criterion.cuda()

    # Data loading code
    data_dir = args.data_dir
    FSSDataset.initialize(args, img_size=crop_size, datapath=data_dir, use_original_imgsize=False)
    train_loader = FSSDataset.build_dataloader(args.benchmark, args.batch_size, args.workers, args.fold, 'trn')
    val_loader = FSSDataset.build_dataloader(args.benchmark, 1, args.workers, args.fold, 'test')
    optimizer = torch.optim.SGD(single_model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0
    labels = []
    path = 'label_files/fewshot_{}.txt'.format(args.benchmark)
    assert os.path.exists(path), '*** Error : {} not exist !!!'.format(path)
    f = open(path, 'r')
    lines = f.readlines()
    for line in lines:
        label = line.strip()
        labels.append(label)
    f.close()
    print(labels)
    texts = []
    label = ['others', '']
    for class_i in range(len(labels)):
        label[1] = labels[class_i]
        text = clip.tokenize(label).cuda()
        texts.append(text)
    lmodel, lpreprocess = clip.load("ViT-B/32", device="cuda")
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume,map_location='cpu')
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        # validate(args,val_loader,val_dataset, model, criterion, eval_score=accuracy,texts=texts,lmodel=lmodel)
        validate(args, val_loader, model, criterion, eval_score=accuracy, texts=texts, lmodel=lmodel)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        train(args,train_loader, model, criterion,optimizer, epoch,
             eval_score=accuracy,texts=texts,lmodel=lmodel)
        val_miou, val_fb_iou = validate(args, val_loader, model, criterion, eval_score=accuracy, texts=texts,
                         lmodel=lmodel)
        if type(best_prec1) == int:
            is_best = val_miou > best_prec1
            best_prec1 = max(val_miou, best_prec1)
        else:
            is_best = val_miou > best_prec1.to(val_miou.device)
            best_prec1 = max(val_miou, best_prec1.to(val_miou.device))
        new_dir = os.path.join(args.filename, 'd'+str(args.crop_size)+'_'+ args.arch+ '_f'+str(args.fold) +'_s' + str(args.lr) +'_B'+ str(args.batch_size))
        if not exists(new_dir):
            os.makedirs(new_dir)
        checkpoint_path = new_dir + '/checkpoint_latest.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path,new_dir=new_dir)
        if (epoch + 1) % 1 == 0:
            history_path = new_dir+'/checkpoint_{:03d}.pth.tar'.format(epoch + 1)
            shutil.copyfile(checkpoint_path, history_path)


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def fast_hist(pred, label,ignore):
    if ignore is not None:
        assert torch.logical_and(ignore, label).sum() == 0
        ignore *= 255
        label = label + ignore
        torch.tensor(pred)[label == 255] = 255
    pred=torch.tensor(pred).cuda()
    area_inter, area_pred, area_gt = [], [], []
    for _pred_mask, _gt_mask in zip(pred, label):
        _inter = _pred_mask[_pred_mask == _gt_mask]
        if _inter.size(0) == 0:  # as torch.histc returns error if it gets empty tensor (pytorch 1.5.1)
            _area_inter = torch.tensor([0, 0], device=_pred_mask.device)
        else:
            _area_inter = torch.histc(_inter, bins=2, min=0, max=1)
        area_inter.append(_area_inter)
        area_pred.append(torch.histc(_pred_mask, bins=2, min=0, max=1))
        area_gt.append(torch.histc(_gt_mask, bins=2, min=0, max=1))
    area_inter = torch.stack(area_inter).t()
    area_pred = torch.stack(area_pred).t()
    area_gt = torch.stack(area_gt).t()
    area_union = area_pred + area_gt - area_inter
    beta=area_inter[1]/area_union[1]
    beta_f = area_inter[0] / area_union[0]
    return area_inter, area_union,beta,beta_f


def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)

def test_sp(args, eval_data_loader, model,
            has_gt=True, texts=None, num_labels=None, lmodel=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    eig_dir = args.eig_dir
    averagemeter = AverageMeter2(args, eval_data_loader.dataset)
    START = time.time()
    for iter, batch in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        batch = utils.to_cuda(batch)
        image = batch['query_img']
        label = batch['query_mask']
        class_sample = batch['class_id']
        name = batch['query_name']
        data_time.update(time.time() - end)
        label, ignore = extract_ignore_idx(label, class_sample)
        h, w = image.size()[2:4]

        torch_resize = Resize([512, 512])
        image = torch_resize(image)
        image = image.cuda()
        label = label.cuda()
        ignore = ignore.cuda()
        image_var = torch.autograd.Variable(image)
        _,features = model(image_var)
        text_features = [lmodel.encode_text(texts[class_i]).detach() for class_i in class_sample]
        text_features = text_features[0].unsqueeze(0).to(torch.float32)
        result = torch.einsum('abcd,aeb->aecd', (features, text_features))
        tempsoftmax = nn.LogSoftmax()
        result = tempsoftmax(result)
        final = sum([resize_4d_tensor(result, w, h)])
        pred = final.argmax(axis=1)

        area_inter1, area_union1,beta1,beta_f = fast_hist(pred, label, ignore)

        BETA = []
        BETA_F=[]
        AREA = []
        EVI=[]
        K = 5
        mask_paths = []
        for i in range(0, K):
            evi_mask = str(Path(eig_dir) / f'{name[0]}_k{i}.png')
            mask_paths.append(evi_mask)
            eigenvector_vis = Image.open(evi_mask).convert('L')
            eigenvector_vis = np.array(eigenvector_vis)
            eigenvector_vis = np.expand_dims(eigenvector_vis, 0)
            point_1 = np.where((eigenvector_vis) == 255)
            eigenvector_vis[point_1] = 1
            area_inter, area_union, beta,beta_f = fast_hist(eigenvector_vis.astype(np.int64), label,
                                                     ignore)
            BETA.append(beta)
            BETA_F.append(beta_f)
            AREA.append([area_inter, area_union])
            EVI.append(eigenvector_vis)

        best_k = BETA.index(max(BETA))
        beta2 = max(BETA)
        area_inter2, area_union2 = AREA[best_k]

        if beta1 >= beta2:
            best_area_inter, best_area_union = area_inter1, area_union1
        elif beta1 < beta2:
            best_area_inter, best_area_union = area_inter2, area_union2

        averagemeter.update(best_area_inter, best_area_union, class_sample.cuda())
        best_output = averagemeter.compute_iou()


        logger.info('===> mIoU {mIoU:.3f}'.format(
            mIoU=best_output[0]))
        logger.info('===>  FB_IoU {FB_IoU:.3f}'.format(
            FB_IoU=best_output[2]))

        batch_time.update(time.time() - end)
        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    # if has_gt:  # val
    END = time.time()
    test_miou, test_mious, test_fb_iou = averagemeter.compute_iou()
    logger.info(' '.join('{:.03f}'.format(i) for i in test_mious))
    logger.info('===> mIoU {mIoU:.3f}'.format(mIoU=test_miou))
    logger.info('===>  FB_IoU {FB_IoU:.3f}'.format(FB_IoU=test_fb_iou))
    logger.info('===> TIME {TIME:.3f}'.format(TIME=END - START))
    return test_miou


def resize_4d_tensor(tensor, width, height):
    # tensor_cpu = tensor.cpu().numpy()
    tensor_cpu = tensor.cpu().data.numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILINEAR))
    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return out


def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase
    fold=args.fold
    crop_size=args.crop_size
    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DPTSegmentationModel(2, backbone=args.arch)
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model).cuda()
    data_dir = args.data_dir
    FSSDataset.initialize(args, img_size=crop_size, datapath=data_dir, use_original_imgsize=True)
    test_loader = FSSDataset.build_dataloader(args.benchmark, args.batch_size, args.workers, args.fold, 'test', 0)
    cudnn.benchmark = True
    start_epoch = 0
    labels = []
    path = './label_files/fewshot_{}.txt'.format(args.benchmark)
    assert os.path.exists(path), '*** Error : {} not exist !!!'.format(path)
    f = open(path, 'r')
    lines = f.readlines()
    for line in lines:
        label = line.strip()
        labels.append(label)
    f.close()
    print(labels)
    texts = []
    label = ['others', '']
    for class_i in range(len(labels)):
        label[1] = labels[class_i]
        text = clip.tokenize(label).cuda()
        texts.append(text)
    lmodel, lpreprocess = clip.load("ViT-B/32", device="cuda")
    labels.insert(0, 'others')
    labels = clip.tokenize(labels).cuda()
    labels = lmodel.encode_text(labels).detach()
    Visualizer.initialize(args.visualize)
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # best_prec1 = checkpoint['val_miou']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    miou = test_sp(args, test_loader, model,
                   has_gt=phase or args.with_gt,
                   texts=texts, num_labels=labels, lmodel=lmodel)
    logger.info('miou: %f', miou)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('-d', '--data-dir', default=None)
    parser.add_argument('--eig_dir', default=None)
    parser.add_argument('--filename', default='./output/',type=str)
    parser.add_argument('-c', '--classes', default=2, type=int)
    parser.add_argument('-s', '--crop-size', default=512, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=6, metavar='N',
                        help='input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--load-release', dest='load_rel', default=None)
    parser.add_argument('--phase', default='val')
    parser.add_argument('--random-scale', default=2, type=float)
    parser.add_argument('--random-rotate', default=10, type=int)
    parser.add_argument('--bn-sync', action='store_true')
    parser.add_argument('--ms', action='store_true',
                        help='Turn on multi-scale testing')
    parser.add_argument('--with-gt', action='store_true')
    parser.add_argument('--test-suffix', default='', type=str)
    parser.add_argument('--has_edge_head', default='True')
    parser.add_argument('--save_vis', default='True', type=str)
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--benchmark', default='pascal',choices=['pascal', 'coco'])
    args = parser.parse_args()

    assert args.data_dir is not None

    print(' '.join(sys.argv))
    print(args)

    if args.bn_sync:
        drn.BatchNorm = batchnormsync.BatchNormSync

    return args
args = parse_args()
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
# logging.basicConfig(format=FORMAT)
if args.cmd == 'train':
    logging.basicConfig(format=FORMAT,filename=os.path.join(args.filename, 'd_'+str(args.crop_size)+'_'+ args.arch+ '_f'+str(args.fold) +'_s' + str(args.lr) +'_B'+ str(args.batch_size)+'.log'))
elif args.cmd == 'test':
    logging.basicConfig(format=FORMAT, filename=os.path.join(args.filename + 'd_' + str(args.crop_size) + '_' + args.arch + '_f' + str(args.fold) + '_s' + str(args.lr) + '_B' + str(args.batch_size) + '.log'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    args = parse_args()
    if args.cmd == 'train':
        train_seg(args)
    elif args.cmd == 'test':
        test_seg(args)


if __name__ == '__main__':
    main()
