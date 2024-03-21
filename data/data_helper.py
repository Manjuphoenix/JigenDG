from os.path import join, dirname

import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from PIL import Image
import cv2
import numpy as np
import random
from torch import fft
from PIL import Image


from data import StandardDataset
from data.JigsawLoader import JigsawDataset, JigsawTestDataset, get_split_dataset_info, _dataset_info, JigsawTestDatasetMultiple
from data.concat_dataset import ConcatDataset

mnist = 'mnist'
mnist_m = 'mnist_m'
svhn = 'svhn'
synth = 'synth'
usps = 'usps'

da_setting_dataset = ["mscoco", "mscoco2", "flir", "m3fd", "m3fdvi", "m3fdbus", "m3fdcar", "m3fdlamp", 
                      "m3fdmotorcycle", "m3fdpeople", "m3fdtruck",
                       "valmscoco", "valmscoco2", "cocobicycle", "cocoperson", "cococar", "flirbicycle", "flirperson", "flircar"]
vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
office_datasets = ["amazon", "dslr", "webcam"]
digits_datasets = [mnist, mnist, svhn, usps]
available_datasets = office_datasets + pacs_datasets + vlcs_datasets + digits_datasets + da_setting_dataset
#office_paths = {dataset: "/home/enoon/data/images/office/%s" % dataset for dataset in office_datasets}
#pacs_paths = {dataset: "/home/enoon/data/images/PACS/kfold/%s" % dataset for dataset in pacs_datasets}
#vlcs_paths = {dataset: "/home/enoon/data/images/VLCS/%s/test" % dataset for dataset in pacs_datasets}

class FFTforvarthreebands_lowband_reduce(object):
    def __call__(self, inp):
        inp = inp.cpu().detach()
        low_pass = torch.ones(inp.shape[0], inp.shape[1], inp.shape[2])
        fft_1 = torch.fft.fftshift(torch.fft.fftn(inp))
        mid_x, mid_y = int(fft_1.shape[1]/2),int(fft_1.shape[2]/2)

        pixel_x, pixel_y = torch.meshgrid(torch.arange(fft_1.shape[1]), torch.arange(fft_1.shape[2]))
        distances = torch.sqrt((pixel_x - mid_x) ** 2 + (pixel_y - mid_y) ** 2)
        thresh = inp.shape[1]/6
        midthresh = thresh*2

        condition = distances<midthresh ###SHOUDL BE < FOR THIS TO WORK!!!###
        midcondition = distances<thresh
        condition = condition.unsqueeze(0)
        # print(condition.shape)
        condition = condition.expand(inp.shape[0], inp.shape[1],inp.shape[2])
        
        # condition = condition.unsqueeze(0)
        midcondition = midcondition.unsqueeze(0)
        # print(condition.shape)
        midcondition = midcondition.expand(inp.shape[0], inp.shape[1],inp.shape[2])
        low_pass[midcondition] = 0
        filtered_mid=torch.multiply(fft_1,low_pass)
        absmid_1, angle_1 = torch.abs(filtered_mid), torch.angle(filtered_mid)
        fft_1 = absmid_1*np.exp((1j) * angle_1)
        fft_final = np.fft.ifftn(np.fft.ifftshift(fft_1))
        fft_final = torch.Tensor(fft_final)
        return fft_final

class FFTforvarthreebands_highband_reduce(object):
    def __call__(self, inp):
        # inp = inp.cpu().detach()          # Uncomment this line if you want to add FFT as transformation by passing a batch of data
        # Eg. inp size would be [30, 3, 224, 224] for a batch of 30 images
        # low_pass = torch.zeros(inp.shape[0],inp.shape[1], inp.shape[2], inp.shape[3])
        inp = np.array(inp)
        mid_pass = torch.zeros(inp.shape[0], inp.shape[1], inp.shape[2])    # This is having 3 channel info only and no batch info--------------- 
        # -----------------------since it is used as torch transformation
        # high_pass = torch.ones(inp.shape[0],inp.shape[1], inp.shape[2], inp.shape[3])
        mid_and_low_pass = torch.ones(inp.shape[0], inp.shape[1], inp.shape[2])
        inp = torch.from_numpy(inp)
        fft_1 = fft.fftshift(fft.fftn(inp))
        fftnoshift_1 = fft.fftn(inp)
        mid_x, mid_y = int(fft_1.shape[1]/2),int(fft_1.shape[2]/2)
        pixel_x, pixel_y = torch.meshgrid(torch.arange(fft_1.shape[1]), torch.arange(fft_1.shape[2]))
        distances = torch.sqrt((pixel_x - mid_x) ** 2 + (pixel_y - mid_y) ** 2)
        thresh = inp.shape[1]/6
        midthresh = thresh*2
        condition = distances<midthresh ###SHOUDL BE < FOR THIS TO WORK!!!###
        midcondition = distances<thresh
        condition = condition.unsqueeze(0)
        condition = condition.expand(inp.shape[0], inp.shape[1], inp.shape[2])
        midcondition = midcondition.unsqueeze(0)
        midcondition = midcondition.expand(inp.shape[0], inp.shape[1], inp.shape[2])
        mid_pass[condition] = 1
        mid_and_low_pass = mid_pass
        filtered_mid=torch.multiply(fft_1,mid_and_low_pass)
        absmid_1, angle_1 = torch.abs(filtered_mid), torch.angle(filtered_mid)
        absmid_1 = absmid_1.cpu()
        angle_1 = angle_1.cpu()
        fft_1 = absmid_1*np.exp((1j) * angle_1)
        fft_final = np.fft.ifftn(np.fft.ifftshift(fft_1))
        fft_final = Image.fromarray(fft_final.astype('uint8'), 'RGB')
        return fft_final



class Canny(object):
    def __call__(self, inp):
        inp = np.array(inp)
        t_lower = 50
        t_upper = 200
        aperture_size = 5
        # img = inp.cpu().detach().numpy()
        canny_img = cv2.Canny(inp, t_lower, t_upper, 
                 apertureSize = aperture_size)
        cv2.imwrite("after_canny.jpg", canny_img)
        canny_img =  Image.fromarray(np.uint8(canny_img)).convert('RGB')
        # canny_img_tensor = torch.Tensor(canny_img)
        return canny_img


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


# This returns train and val dataloaders.....       THIS IS FOR FLIR
# def get_train_dataloader(args, patches):

#     name_person_num = 0
#     name_car_num = 0
#     name_bicycle_num = 0
#     tuple_of_filename_class_list = []

#     dataset_list = args.source
#     assert isinstance(dataset_list, list)
#     datasets = []
#     val_datasets = []
#     img_transformer, tile_transformer = get_train_transformers(args)
#     limit = args.limit_source
#     for dname in dataset_list:
#         name_train, name_val, labels_train, labels_val = get_split_dataset_info(os.path.join(dirname(__file__), 'txt_lists/')+ dname +'_train.txt', args.val_size)
#         for name_train_i in name_train:
#             if 'person' in name_train_i:
#                 name_person_num +=1
#                 tuple_of_filename_class_list.append((name_train_i,2))
#             if 'car' in name_train_i:
#                 tuple_of_filename_class_list.append((name_train_i,1))
#                 name_car_num +=1
#             if 'bicycle' in name_train_i:
#                 tuple_of_filename_class_list.append((name_train_i,0))
#                 name_bicycle_num +=1
#         train_dataset = JigsawDataset(name_train, labels_train, patches=patches, img_transformer=img_transformer,
#                                       tile_transformer=tile_transformer, jig_classes=args.jigsaw_n_classes, bias_whole_image=args.bias_whole_image)
#         if limit:
#             train_dataset = Subset(train_dataset, limit)
#         datasets.append(train_dataset)
#         val_datasets.append(
#             JigsawTestDataset(name_val, labels_val, img_transformer=get_val_transformer(args),
#                               patches=patches, jig_classes=args.jigsaw_n_classes))
    
#     print("bicycle", name_bicycle_num)
#     print("car", name_car_num)
#     print("name_person_num", name_person_num)
#     dataset = ConcatDataset(datasets)
#     val_dataset = ConcatDataset(val_datasets)


def get_train_dataloader(args, patches):
    name_bus_num = 0
    name_car_num = 0
    name_lamp_num = 0
    name_motor_num = 0
    name_people_num = 0
    name_truck_num = 0
    tuple_of_filename_class_list = []
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    img_transformer, tile_transformer = get_train_transformers(args)
    limit = args.limit_source
    train_list = []
    for dname in dataset_list:
        name_train, name_val, labels_train, labels_val = get_split_dataset_info(os.path.join(dirname(__file__), 'txt_lists/')+ dname +'_train.txt', args.val_size)
        # print(name_train[0], "-------------------", name_val[0])
        # print(HEY)
        for name_train_i in name_train:
            if 'bus' in name_train_i:
                name_bus_num +=1
                tuple_of_filename_class_list.append((name_train_i,0))
            if 'car' in name_train_i:
                tuple_of_filename_class_list.append((name_train_i,1))
                name_car_num +=1
            if 'lamp' in name_train_i:
                tuple_of_filename_class_list.append((name_train_i,2))
                name_lamp_num +=1
            if 'motorcycle' in name_train_i:
                name_motor_num +=1
                tuple_of_filename_class_list.append((name_train_i,3))
            if 'people' in name_train_i:
                tuple_of_filename_class_list.append((name_train_i,4))
                name_people_num +=1
            if 'truck' in name_train_i:
                tuple_of_filename_class_list.append((name_train_i,5))
                name_truck_num +=1
        train_dataset = JigsawDataset(name_train, labels_train, patches=patches, img_transformer=img_transformer,
                                      tile_transformer=tile_transformer, jig_classes=args.jigsaw_n_classes, bias_whole_image=args.bias_whole_image)
        if limit:
            train_dataset = Subset(train_dataset, limit)
        datasets.append(train_dataset)
        val_datasets.append(
            JigsawTestDataset(name_val, labels_val, img_transformer=get_val_transformer(args),
                              patches=patches, jig_classes=args.jigsaw_n_classes))
    
    print("bus", name_bus_num)
    print("car", name_car_num)
    print("lamp", name_lamp_num)
    print('motorcycle', name_motor_num)
    print('people', name_people_num)
    print('truck', name_truck_num)
    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)

    # breakpoint()
    # weight = make_weight_for_balanced_classes(tuple_of_filename_class_list, len(['bicycle', 'car', 'person']))  # for flir

    weight = make_weight_for_balanced_classes(tuple_of_filename_class_list, len(['bus', 'car', 'lamp', 'motorcycle', 'people', 'truck']))   # for m3fd
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
    
    # if args.random_horiz_flip > 0.0:
    #     img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    # if args.jitter > 0.0:
    #     img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))

    tile_tr = [transforms.ToTensor()]
    # if args.tile_random_grayscale:
    #     tile_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    # tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    # tile_tr.append(transforms.ToTensor())

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)


    
def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)


def get_target_jigsaw_loader(args):
    img_transformer, tile_transformer = get_train_transformers(args)
    # img_transformer.transforms.insert(1, FFTforvarthreebands_highband_reduce())
    # tile_transformer.transforms.insert(1, FFTforvarthreebands_highband_reduce())
    name_train, _, labels_train, _ = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % args.target), 0)
    dataset = JigsawDataset(name_train, labels_train, patches=False, img_transformer=img_transformer,
                            tile_transformer=tile_transformer, jig_classes=args.jigsaw_n_classes, bias_whole_image=args.bias_whole_image)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    return loader