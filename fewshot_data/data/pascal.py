r""" PASCAL-5i few-shot semantic segmentation dataset """
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'train'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        self.base_path = os.path.join(datapath, 'VOC2012/')
        self.img_path = os.path.join(self.base_path, 'JPEGImages/')
        self.ann_path = os.path.join(self.base_path, 'SegmentationClassAug/')
        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
    #
    def __len__(self):
        return len(self.img_metadata) if self.split == 'train' else 1000

    def __getitem__(self, idx):
        # data = [Image.open(join(self.data_dir, self.image_list[index]))]
        data = []
        idx %= len(self.img_metadata)
        query_name, class_sample = self.img_metadata[idx]
        query = os.path.join(self.img_path, query_name) + '.jpg'
        # query_id = query.split('/')[-1].split('.')[0]
        mask = Image.open(os.path.join(self.ann_path, query_name) + '.png')
        mask_name = os.path.join(self.ann_path, query_name) + '.png'
        query = Image.open(query).convert('RGB')
        data.append(query)
        data.append(mask)
        data = list(self.transform(*data))
        _edgemap = data[1].clone()
        _edgemap = _edgemap.numpy()
        _edgemap = self.mask_to_onehot(_edgemap, self.nclass)
        _edgemap = self.onehot_to_binary_edges(_edgemap, 2, self.nclass)
        edgemap = torch.from_numpy(_edgemap).float()
        data.append(edgemap)
        batch = {'query_img': data[0],
                 'query_mask': data[1],
                 'edge_gts': data[2],
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
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        if self.split == 'train':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):
        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join(self.base_path + '%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'train':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            # img_metadata = read_metadata(self.phase, self.fold)
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise
    #     query_img, query_cmask, class_sample, org_qry_imsize,query_name = self.load_frame()
    #     query_img = self.transform(query_img)
    #     if not self.use_original_imgsize:
    #         query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img[0].size()[-2:], mode='nearest').squeeze()
    #     query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)
    #     batch = {'query_img': query_img,
    #              'query_mask': query_mask,
    #              'query_name': query_name,
    #              'query_ignore_idx': query_ignore_idx,
    #              'class_id': torch.tensor(class_sample)}
    #
    #     return batch
    #
    #
    # def extract_ignore_idx(self, mask, class_id):
    #     boundary = (mask / 255).floor()
    #     mask[mask != class_id + 1] = 0
    #     mask[mask == class_id + 1] = 1
    #
    #     return mask, boundary
    #
    # def load_frame(self):
    #     class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
    #     query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
    #     query_img = self.read_img(query_name)
    #     query_mask = self.read_mask(query_name)
    #
    #
    #     org_qry_imsize = query_img.size
    #
    #     return query_img, query_mask,class_sample, org_qry_imsize,query_name
    #
    # def read_mask(self, img_name):
    #     r"""Return segmentation mask in PIL Image"""
    #     mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
    #     return mask
    #
    # def read_img(self, img_name):
    #     r"""Return RGB image in PIL Image"""
    #     return Image.open(os.path.join(self.img_path, img_name) + '.jpg')
    #
    # def sample_episode(self, idx):
    #     query_name, class_sample = self.img_metadata[idx]
    #
    #     support_names = []
    #     while False:
    #     # while True:  # keep sampling support set if query == support
    #         support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
    #         if query_name != support_name: support_names.append(support_name)
    #         if len(support_names) == self.shot: break
    #
    #     return query_name, support_names, class_sample
    #
    # def build_class_ids(self):
    #     nclass_trn = self.nclass // self.nfolds
    #     class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
    #     class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
    #
    #     if self.split == 'trn':
    #         return class_ids_trn
    #     else:
    #         if cross:
    #             class_ids_val = list(range(20))
    #             return class_ids_val
    #         else:
    #             return class_ids_val
    #
    # def build_img_metadata(self):
    #
    #     def read_metadata(split, fold_id):
    #         fold_n_metadata = os.path.join('./fewshot_data/data/splits/pascal/%s/fold%d.txt' % (split, fold_id))
    #         with open(fold_n_metadata, 'r') as f:
    #             fold_n_metadata = f.read().split('\n')[:-1]
    #         fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
    #         return fold_n_metadata
    #
    #     img_metadata = []
    #     if self.split == 'trn':  # For training, read image-metadata of "the other" folds
    #         for fold_id in range(self.nfolds):
    #             if fold_id == self.fold:  # Skip validation fold
    #                 continue
    #             img_metadata += read_metadata(self.split, fold_id)
    #     elif self.split == 'val':  # For validation, read image-metadata of "current" fold
    #         if cross:
    #             for fold_id in range(self.nfolds):
    #                 img_metadata += read_metadata(self.split, fold_id)
    #         else:
    #             img_metadata = read_metadata(self.split, self.fold)
    #     else:
    #         raise Exception('Undefined split %s: ' % self.split)
    #
    #     print('Total (%s) images are : %d' % (self.split, len(img_metadata)))
    #
    #     return img_metadata
    #
    # def build_img_metadata_classwise(self):
    #     img_metadata_classwise = {}
    #     for class_id in range(self.nclass):
    #         img_metadata_classwise[class_id] = []
    #
    #     for img_name, img_class in self.img_metadata:
    #         img_metadata_classwise[img_class] += [img_name]
    #     return img_metadata_classwise
