r""" Dataloader builder for few-shot semantic segmentation dataset  """
# from torchvision import transforms
from torch.utils.data import DataLoader
import data_transforms as transforms
from fewshot_data.data.pascal import DatasetPASCAL
from fewshot_data.data.coco import DatasetCOCO
from fewshot_data.data.fss import DatasetFSS
import json
from os.path import exists, join, split
class FSSDataset:

    @classmethod
    def initialize(cls, args,img_size, datapath, use_original_imgsize):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        # cls.datapath_test = datapath_test
        cls.use_original_imgsize = use_original_imgsize

        # cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
        #                                     transforms.ToTensor(),
        #                                     transforms.Normalize(cls.img_mean, cls.img_std)])

        info = json.load(open(join(datapath, 'info.json'), 'r'))
        normalize = transforms.Normalize(mean=info['mean'],
                                             std=info['std'])
        # normalize = transforms.Normalize(mean=cls.img_mean,
        #                                  std=cls.img_std)
        t = []
        if args.random_rotate > 0:
            t.append(transforms.RandomRotate(args.random_rotate))
        if args.random_scale > 0:
            t.append(transforms.RandomScale(args.random_scale))
        t.extend([transforms.RandomCrop(args.crop_size),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  normalize])
        cls.trn_transform = transforms.Compose(t)
        cls.val_transform  = transforms.Compose([
            transforms.RandomCrop(img_size),
            transforms.ToTensor(),
            normalize])
        cls.transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])



    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=0):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0
        if split == 'trn':
            dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.trn_transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
            dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)
        elif split=='val':
            dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.val_transform, split=split,
                                              shot=shot, use_original_imgsize=cls.use_original_imgsize)
            dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)
        else:
            dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split,
                                              shot=shot, use_original_imgsize=cls.use_original_imgsize)
            dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader
