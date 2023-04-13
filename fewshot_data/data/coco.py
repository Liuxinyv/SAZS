r""" COCO-20i few-shot semantic segmentation dataset """
import os
import pickle

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from rebuttle.adjust import *
class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.split_coco = split if split == 'val2014' else 'train2014'
        self.base_path = os.path.join(datapath, 'COCO2014')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.has_edge_head = True
        self.class_ids,self.class_ids_val = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()
        data = []
        data.append(query_img)
        data.append(query_mask)
        data = list(self.transform(*data))
        _edgemap = data[1].clone()
        _edgemap = _edgemap.numpy()
        _edgemap = self.mask_to_onehot(_edgemap, self.nclass)
        _edgemap = self.onehot_to_binary_edges(_edgemap, 2, self.nclass)
        edgemap = torch.from_numpy(_edgemap).float()
        data.append(edgemap)
        batch = {'query_img': data[0],
                 'query_mask': data[1],
                 'edge_gts':data[2],
                 'query_name': query_name,
                     'class_id': torch.tensor(class_sample)}

        return batch

    def mask_to_onehot(self, mask, num_classes):
        _mask = [mask == (i) for i in range(num_classes)]
        return np.array(_mask).astype(np.uint8)

    def onehot_to_binary_edges(self, mask, radius, num_classes):
        mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
        edgemap = np.zeros(mask.shape[1:])
        for i in range(num_classes):
            dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
            dist = dist[1:-1, 1:-1]
            dist[dist > radius] = 0
            edgemap += dist
        # edgemap = np.expand_dims(edgemap, axis=0)
        edgemap = (edgemap > 0).astype(np.uint8)
        return edgemap
    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        # class_ids = class_ids_trn if self.split == 'trn' else class_ids_val
        if cross:
            class_ids=list(range(80))
            return class_ids, class_ids_val
        else:
            class_ids = class_ids_trn if self.split == 'trn' else class_ids_val
            return class_ids,class_ids_val

    def build_img_metadata_classwise(self):
        # with open('./fewshot_data/data/splits/coco/%s/fold%d.pkl' % (self.split, self.fold), 'rb') as f:
        with open(os.path.join(self.base_path + '/coco/' + '%s/fold%d.pkl' % (self.split, self.fold)), 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        if cross:
            with open(os.path.join(self.base_path + 'coco/' + '%s/fold%d.pkl' % (self.split, 1)), 'rb') as f:
                img_metadata_classwise_add = pickle.load(f)
            for i in self.class_ids_val:
                img_metadata_classwise[i]=img_metadata_classwise_add[i]
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = Image.open(mask_path[:mask_path.index('.jpg')] + '.png')
        return mask

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        query_mask = self.read_mask(query_name)

        org_qry_imsize = query_img.size

        # query_mask[query_mask != class_sample + 1] = 0
        # query_mask[query_mask == class_sample + 1] = 1

        support_names = []
        if self.shot:
            while True:  # keep sampling support set if query == support
                support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
                if query_name != support_name: support_names.append(support_name)
                if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        if self.shot:
            for support_name in support_names:
                support_imgs.append(Image.open(os.path.join(self.base_path, support_name)).convert('RGB'))
                support_mask = self.read_mask(support_name)
                support_mask[support_mask != class_sample + 1] = 0
                support_mask[support_mask == class_sample + 1] = 1
                support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize

