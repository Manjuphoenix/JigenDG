import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2
from random import sample, random


def get_random_subset(names, labels, percent):
    """

    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val


def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


def get_split_dataset_info(txt_list, val_percentage):
    names, labels = _dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)


class JigsawDataset(data.Dataset):
    def __init__(self, names, labels, jig_classes=100, img_transformer=None, tile_transformer=None, patches=True, bias_whole_image=None):
        self.data_path = ""
        self.names = names
        self.labels = labels

        self.N = len(self.names)
        self.permutations = self.__retrieve_permutations(jig_classes)
        self.grid_size = 3
        self.bias_whole_image = bias_whole_image
        if patches:         # this wont happen..........
            self.patch_size = 64
        self._image_transformer = img_transformer
        self._augment_tile = tile_transformer
        if patches:
            self.returnFunc = lambda x: x
        else:
            def make_grid(x):
                return torchvision.utils.make_grid(x, self.grid_size, padding=0)
            self.returnFunc = make_grid

    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size             # 222/3 i.e 74
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        # print(img.size[0], n, self.grid_size, w, y, x, [x * w, y * w, (x + 1) * w, (y + 1) * w], tile.shape, "++++++++++++++++++++")
        return tile
    
    def get_image(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        tmp = np.asarray(img)
        # cv2.imwrite("try_org.png", tmp)
        return self._image_transformer(img)
    

####################### new loader i.e, JUNK..............####################
    # def get_image(self, indexes):
    #     imgs = []
    #     for i in indexes:
    #         framename = self.data_path + '/' + self.names[i]
    #         img = Image.open(framename).convert('RGB')
    #         tmp = np.asarray(img)
    #         imgs.append(tmp)            # list of numpy arrays which will be our images
    #     # cv2.imwrite("try_org.png", tmp)
    #     print(imgs, "************************88")
    #     return self._image_transformer(imgs)
    

        
    def __getitem__(self, index):
        img = self.get_image(index)
        n_grids = self.grid_size ** 2       # n_grids will be 9
        tiles = [None] * n_grids            # tiles will have a list of len 9 with all none eg. [None, None, None, None, None, None, None, None, None]
        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)

        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        # order = 0         # for verification
        # print(order, "****************")         # for verification
        # order = 1
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0
        if order == 0:
            # print(HWY)
            data = tiles
            # print(len(tiles), tiles[1].shape)           # len is 9 and shape is torch.Size([3, 74, 74])
        else:
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]
            # breakpoint()
        # print(data[0].shape)
        data = torch.stack(data, 0)
        cv2.imwrite("data_0.png", data[0].cpu().detach().permute(1, 2, 0).numpy()*255)
        cv2.imwrite("data_1.png", data[1].cpu().detach().permute(1, 2, 0).numpy()*255)
        cv2.imwrite("data_2.png", data[2].cpu().detach().permute(1, 2, 0).numpy()*255)
        cv2.imwrite("data_3.png", data[3].cpu().detach().permute(1, 2, 0).numpy()*255)
        cv2.imwrite("data_4.png", data[4].cpu().detach().permute(1, 2, 0).numpy()*255)
        cv2.imwrite("data_5.png", data[5].cpu().detach().permute(1, 2, 0).numpy()*255)
        cv2.imwrite("data_6.png", data[6].cpu().detach().permute(1, 2, 0).numpy()*255)
        cv2.imwrite("data_7.png", data[7].cpu().detach().permute(1, 2, 0).numpy()*255)
        cv2.imwrite("data_8.png", data[8].cpu().detach().permute(1, 2, 0).numpy()*255)
        # print(self.returnFunc(data), int(order), int(self.labels[index]), "-------------------")
        print(HEY)
        return self.returnFunc(data), int(order), int(self.labels[index])

    def __len__(self):
        return len(self.names)

    def __retrieve_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm


class JigsawTestDataset(JigsawDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        # img = cv2.imread(framename)
        img = Image.open(framename).convert('RGB')
        # pil_img = Image.fromarray(np.uint8(framename)).convert('RGB')
        # img = np.array(img)
        # print(img)
        # print(type(img))
        # breakpoint()
        # print(HEY)
        return self._image_transformer(img), 0, int(self.labels[index])


class JigsawTestDatasetMultiple(JigsawDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        self._image_transformer = transforms.Compose([
            transforms.Resize(255, Image.BILINEAR),
        ])
        self._image_transformer_full = transforms.Compose([
            transforms.Resize(225, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._augment_tile = transforms.Compose([
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        _img = Image.open(framename).convert('RGB')
        img = self._image_transformer(_img)

        w = float(img.size[0]) / self.grid_size
        n_grids = self.grid_size ** 2
        images = []
        jig_labels = []
        tiles = [None] * n_grids
        for n in range(n_grids):
            y = int(n / self.grid_size)
            x = n % self.grid_size
            tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
            tile = self._augment_tile(tile)
            tiles[n] = tile
        for order in range(0, len(self.permutations)+1, 3):
            if order==0:
                data = tiles
            else:
                data = [tiles[self.permutations[order-1][t]] for t in range(n_grids)]
            data = self.returnFunc(torch.stack(data, 0))
            images.append(data)
            jig_labels.append(order)
        images = torch.stack(images, 0)
        jig_labels = torch.LongTensor(jig_labels)
        return images, jig_labels, int(self.labels[index])
