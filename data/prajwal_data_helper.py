from os.path import join, dirname

import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision


from data import StandardDataset
from data.JigsawLoader import JigsawDataset, JigsawTestDataset, get_split_dataset_info, _dataset_info, JigsawTestDatasetMultiple
from data.concat_dataset import ConcatDataset

mnist = 'mnist'
mnist_m = 'mnist_m'
svhn = 'svhn'
synth = 'synth'
usps = 'usps'

da_setting_dataset = ["mscoco", "flir", "valmscoco"]
vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
office_datasets = ["amazon", "dslr", "webcam"]
digits_datasets = [mnist, mnist, svhn, usps]
available_datasets = office_datasets + pacs_datasets + vlcs_datasets + digits_datasets + da_setting_dataset
#office_paths = {dataset: "/home/enoon/data/images/office/%s" % dataset for dataset in office_datasets}
#pacs_paths = {dataset: "/home/enoon/data/images/PACS/kfold/%s" % dataset for dataset in pacs_datasets}
#vlcs_paths = {dataset: "/home/enoon/data/images/VLCS/%s/test" % dataset for dataset in pacs_datasets}


def make_weight_for_balanced_classes(images, nclasses):
    count = [0]*nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.]*nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0]*len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight
#paths = {**office_paths, **pacs_paths, **vlcs_paths}

dataset_std = {mnist: (0.30280363, 0.30280363, 0.30280363),
               mnist_m: (0.2384788, 0.22375608, 0.24496263),
               svhn: (0.1951134, 0.19804622, 0.19481073),
               synth: (0.29410212, 0.2939651, 0.29404707),
               usps: (0.25887518, 0.25887518, 0.25887518),
               }

dataset_mean = {mnist: (0.13909429, 0.13909429, 0.13909429),
                mnist_m: (0.45920207, 0.46326601, 0.41085603),
                svhn: (0.43744073, 0.4437959, 0.4733686),
                synth: (0.46332872, 0.46316052, 0.46327512),
                usps: (0.17025368, 0.17025368, 0.17025368),
                }


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def get_train_dataloader(args, patches):

    name_person_num = 0
    name_car_num = 0
    name_bicycle_num = 0
    tuple_of_filename_class_list = []

    dataset_list = args.source
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    img_transformer, tile_transformer = get_train_transformers(args)
    limit = args.limit_source
    for dname in dataset_list:
        print(dname)
        # breakpoint()
        name_train, name_val, labels_train, labels_val = get_split_dataset_info(os.path.join(dirname(__file__), 'txt_lists/')+ dname +'_train.txt', args.val_size)
        print(name_train)
        for name_train_i in name_train:
            if 'person' in name_train_i:
                name_person_num +=1
                tuple_of_filename_class_list.append((name_train_i,2))
            if 'car' in name_train_i:
                tuple_of_filename_class_list.append((name_train_i,1))
                name_car_num +=1
            if 'bicycle' in name_train_i:
                tuple_of_filename_class_list.append((name_train_i,0))
                name_bicycle_num +=1
        train_dataset = JigsawDataset(name_train, labels_train, patches=patches, img_transformer=img_transformer,
                                      tile_transformer=tile_transformer, jig_classes=args.jigsaw_n_classes, bias_whole_image=args.bias_whole_image)
        print(len(train_dataset), "dataset l")
        if limit:
            train_dataset = Subset(train_dataset, limit)
        datasets.append(train_dataset)
        val_datasets.append(
            JigsawTestDataset(name_val, labels_val, img_transformer=get_val_transformer(args),
                              patches=patches, jig_classes=args.jigsaw_n_classes))
    
    print("bicycle", name_bicycle_num)
    print("car", name_car_num)
    print("name_person_num", name_person_num)
    dataset = ConcatDataset(datasets)
    print(len(dataset), "l of dataset")
    val_dataset = ConcatDataset(val_datasets)

    pre_process = transforms.Compose([transforms.Resize((222, 222)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                        mean=(0.5776, 0.5776, 0.5776),
                                        std=(0.1319, 0.1319, 0.1319))])
    mscoco_dataset = torchvision.datasets.ImageFolder(root="/home/wirin/manjunath/uda/dataset_dir/uda_data/mscoco2/val/",
                                        transform=pre_process)
    print(len(mscoco_dataset), "mcoco dataset len for sampler")
    print(mscoco_dataset.imgs[:10])
    # print(mscoco_dataset.classes)
    print(type(mscoco_dataset.imgs))
    print(tuple_of_filename_class_list[:10])
    # print(hye)
    # breakpoint()
    weight = make_weight_for_balanced_classes(mscoco_dataset.imgs, mscoco_dataset.classes)
    weight=torch.DoubleTensor(weight)

    sampleresh = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler= sampleresh, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader



def get_val_dataloader(args, patches=False):
    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', '%s_test.txt' % args.target))
    img_tr = get_val_transformer(args)
    val_dataset = JigsawTestDataset(names, labels, patches=patches, img_transformer=img_tr, jig_classes=args.jigsaw_n_classes)
    if args.limit_target and len(val_dataset) > args.limit_target:
        val_dataset = Subset(val_dataset, args.limit_target)
        print("Using %d subset of val dataset" % args.limit_target)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader


def get_jigsaw_val_dataloader(args, patches=False):
    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', '%s_test.txt' % args.target))
    img_tr = [transforms.Resize((args.image_size, args.image_size))]
    tile_tr = [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    img_transformer = transforms.Compose(img_tr)
    tile_transformer = transforms.Compose(tile_tr)
    val_dataset = JigsawDataset(names, labels, patches=patches, img_transformer=img_transformer,
                                      tile_transformer=tile_transformer, jig_classes=args.jigsaw_n_classes, bias_whole_image=args.bias_whole_image)
    if args.limit_target and len(val_dataset) > args.limit_target:
        val_dataset = Subset(val_dataset, args.limit_target)
        print("Using %d subset of val dataset" % args.limit_target)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader


def get_train_transformers(args):
    # img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale)), transforms.Grayscale(num_output_channels=3)]
    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]
    
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))

    tile_tr = []
    if args.tile_random_grayscale:
        tile_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)


def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)


def get_target_jigsaw_loader(args):
    img_transformer, tile_transformer = get_train_transformers(args)
    name_train, _, labels_train, _ = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % args.target), 0)
    dataset = JigsawDataset(name_train, labels_train, patches=False, img_transformer=img_transformer,
                            tile_transformer=tile_transformer, jig_classes=args.jigsaw_n_classes, bias_whole_image=args.bias_whole_image)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    return loader