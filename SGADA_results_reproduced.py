import argparse

import torch
from IPython.core.debugger import set_trace
from torch import nn
import torch.nn.functional as func
import torchvision
from torchvision import datasets, transforms
from data import data_helper
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
# from utils.Logger import Logger
import numpy as np
from sklearn.metrics import accuracy_score
import random
import os
import itertools
import cv2
import time
from torchvision import models
# from torchmetrics import Accuracy as acc

seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=225, help="Image size")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", type=float, default=0.0, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", type=float, default=0.0, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", type=float, default=0.1, help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int, help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int, help="If set, it will limit the number of testing samples")

    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=15, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=31, help="Number of classes")
    parser.add_argument("--jigsaw_n_classes", "-jc", type=int, default=100, help="Number of classes for the jigsaw task")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="caffenet")
    parser.add_argument("--jig_weight", type=float, default=0.1, help="Weight for the jigsaw puzzle")
    parser.add_argument("--target_weight", type=float, default=0, help="Weight for target jigsaw task")
    parser.add_argument("--entropy_weight", type=float, default=0, help="Weight for target entropy")
    
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=None, type=float, help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool,
                        help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=False, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")

    return parser.parse_args()



def get_logger(log_file):
    from logging import getLogger, FileHandler, StreamHandler, Formatter, DEBUG, INFO  # noqa
    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    sh = StreamHandler()
    sh.setLevel(INFO)
    for handler in [fh, sh]:
        formatter = Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
    logger = getLogger('adda')
    logger.setLevel(INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
    

def entropy_loss(x):
    return torch.sum(-func.softmax(x, 1) * func.log_softmax(x, 1), 1).mean()


# Lu Gan Dataloader....
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


def get_mscoco(dataset_root, batch_size, train):
    """Get MSCOCO datasets loader

    Args:
        dataset_root (str): path to the dataset folder
        batch_size (int): batch size
        train (bool): create loader for training or test set

    Returns:
        obj: dataloader object for MSCOCO dataset
    """  
    if train:
        pre_process = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4017, 0.3791, 0.3656), std=(0.2093, 0.2019, 0.1996))
        ])
        mscoco_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'uda_data/mscoco2/train'),
                                             transform=pre_process)

        weight = make_weight_for_balanced_classes(mscoco_dataset.imgs, len(mscoco_dataset.classes))
        weight=torch.DoubleTensor(weight)

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))
        mscoco_data_loader = torch.utils.data.DataLoader(
            dataset=mscoco_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    else:
        pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              mean=(0.3961, 0.3743, 0.3603),
                                              std=(0.2086, 0.2012, 0.1987))])
        mscoco_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'uda_data/mscoco2/val'),
                                            transform=pre_process)

        mscoco_data_loader = torch.utils.data.DataLoader(
            dataset=mscoco_dataset,
            batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    return mscoco_data_loader


def get_m3fd(dataset_root, batch_size, train, test=False, pseudo_label=False):
    """Get M3FD datasets loader

    Args:
        dataset_root (str): path to the dataset folder
        batch_size (int): batch size
        train (bool): create loader for training or test set

    Returns:
        obj: dataloader object for M3FD dataset
    """ 
    # dataset and data loader
    if train:
        pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.4821, 0.4821, 0.4821),
                                          std=(0.2081, 0.2081, 0.2081))])
        m3fd_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'uda_data/m3fd/train'),
                                             transform=pre_process)
        weight = make_weight_for_balanced_classes(m3fd_dataset.imgs, len(m3fd_dataset.classes))
        weight=torch.DoubleTensor(weight)

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))

        m3fd_data_loader = torch.utils.data.DataLoader(
            dataset=m3fd_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    else:
        pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.4810, 0.4810, 0.4810),
                                          std=(0.2081, 0.2081, 0.2081))])
        if test:
            m3fd_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'uda_data/m3fd/test'),
                                                transform=pre_process)
            m3fd_data_loader = torch.utils.data.DataLoader(
                dataset=m3fd_dataset,
                batch_size=1,
                shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        else:
            m3fd_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'uda_data/m3fd/val'),
                                                transform=pre_process)
            m3fd_data_loader = torch.utils.data.DataLoader(
                dataset=m3fd_dataset,
                batch_size=batch_size,
                shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    return m3fd_data_loader


def get_flir(dataset_root, batch_size, train, pseudo_label=False):
    """Get FLIR datasets loader

    Args:
        dataset_root (str): path to the dataset folder
        batch_size (int): batch size
        train (bool): create loader for training or test set

    Returns:
        obj: dataloader object for FLIR dataset
    """ 

    # dataset and data loader
    if train:
        pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5776, 0.5776, 0.5776),
                                          std=(0.1319, 0.1319, 0.1319))])
        flir_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'uda_data/flir/train'),
                                             transform=pre_process)
        weight = make_weight_for_balanced_classes(flir_dataset.imgs, len(flir_dataset.classes))
        weight=torch.DoubleTensor(weight)

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))

        flir_data_loader = torch.utils.data.DataLoader(
            dataset=flir_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    else:
        pre_process = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=(0.5587, 0.5587, 0.558),
                                          std=(0.1394, 0.1394, 0.1394))])
        flir_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'uda_data/flir/val'),
                                            transform=pre_process)
        flir_data_loader = torch.utils.data.DataLoader(
            dataset=flir_dataset,
            batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    return flir_data_loader




class ResNet50Mod(nn.Module):
    def __init__(self):
        super(ResNet50Mod, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.stage1 = nn.Sequential(
            model_resnet50.conv1,
            model_resnet50.bn1,
            model_resnet50.relu,
            model_resnet50.maxpool
        )
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        in_features = model_resnet50.fc.in_features
        self.fc = nn.Linear(2048, 3)
    
    def forward(self, x):
        x = self.stage1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        model = model_factory.get_network(args.network)(jigsaw_classes=args.jigsaw_n_classes + 1, classes=args.n_classes)
        # model = ResNet50Mod()
        self.model = model.to(device)
        self.data_root = '/home/wirin/manjunath/uda/dataset_dir'
        # print(self.model)
        if args.target in args.source:
            print("No need to include target in source, it is automatically done by this script")
            k = args.source.index(args.target)
            args.source = args.source[:k] + args.source[k + 1:]
            print("Source: %s" % args.source)
        # self.source_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
        # self.val_loader = data_helper.get_train_val_dataloader(args, patches=model.is_patch_based())
        # self.target_jig_loader = data_helper.get_target_jigsaw_loader(args)
        # self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
        self.source_loader = get_mscoco(self.data_root, batch_size=32, train=True)
        self.val_loader = get_mscoco(self.data_root, batch_size=32, train=False)
        self.target_loader = get_m3fd(self.data_root, batch_size=32, train=False)

        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        # print("Dataset size: train %d, target jig: %d, val %d, test %d" % (
        #     len(self.source_loader.dataset), len(self.target_jig_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        print("Train len", len(self.source_loader.dataset), "Val len", len(self.val_loader.dataset), "Test len", len(self.target_loader.dataset))
        self.optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.5, 0.999), lr=5e-4, weight_decay=2.5e-5)
        # self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all, nesterov=args.nesterov)
        self.jig_weight = args.jig_weight
        self.target_weight = args.target_weight
        self.target_entropy = args.entropy_weight
        self.only_non_scrambled = args.classify_only_sane
        self.n_classes = args.n_classes


    def _do_epoch(self, logger):
        criterion = nn.CrossEntropyLoss()
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total_params: {}".format(pytorch_total_params))
        self.model.train()
        for it, (source_batch) in enumerate(self.source_loader):
            data, class_l = source_batch
            data, class_l = data.to(self.device), class_l.to(self.device)
            self.optimizer.zero_grad()
            # _, class_logit = self.model(data)
            class_logit = self.model(data)
            class_loss = criterion(class_logit, class_l)
            string_class_loss  = str(class_loss.item())
            if it%10==0:
                logger.info("Epoch no: " + str(self.current_epoch) + "," + " class_loss " + string_class_loss)
            class_loss.backward()
            self.optimizer.step()

        
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': class_loss,
            # Add other relevant information if needed
            }, str(self.args.folder_name)+'/epoch'+str(self.current_epoch)+".pth")


        del class_loss


        # model = self.model
        # checkpoint = torch.load("./outputs/source-only-april15-adam-lr-weight-decay-stride-run2/epoch8.pth")
        # model.load_state_dict(checkpoint['model_state_dict'])


        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct = self.do_test(loader, logger, phase)

                class_acc = float(class_correct) / total
                # self.logger.log_test(phase, {"jigsaw": jigsaw_acc, "class": class_acc})
                self.results[phase][self.current_epoch] = class_acc


    def do_test(self, loader, logger, phase):

        class_correct = torch.zeros(10)
        class_total = torch.zeros(10)

        jigsaw_correct = 0
        class_correct = 0
        total_loss = 0
        total_cls_loss = 0.0
        total_cls_acc = 0.0
        classes = ["bicycle", "car", "person"]
        class_correct_list = list(0. for i in range(3))
        class_total = list(0. for i in range(3))
        c = []
        test_predicted = torch.empty(32)
        targets, probas = [], []
        
        if phase=="test":       # This is the test dataloader part
            for it, (data, class_l) in enumerate(loader):
                data, class_l = data.to(self.device), class_l.to(self.device)
                # _, class_logit = self.model(data)
                class_logit = self.model(data)
                probabilities = torch.softmax(class_logit, dim=1)
                # breakpoint()

                _, test_predicted = torch.max(class_logit, 1)

                _, test_predicted_new = torch.max(probabilities, 1)
                max_prob, _ = torch.max(probabilities, 1)
                # test_predicted_new[max_prob < 0.5] = -1
                targets.extend(class_l.cpu().numpy().tolist())
                probas.extend(test_predicted_new.cpu().numpy().tolist())

                c = (test_predicted==class_l).squeeze()
                for i in range(data.size(0)):
                    label = class_l[i]
                    class_correct_list[label] += c[i].item()
                    class_total[label] += 1
                criterion = nn.CrossEntropyLoss()
                class_loss = criterion(class_logit, class_l)
                total_loss += class_loss.item()

                _, cls_pred = class_logit.max(dim=1)
                class_correct += torch.sum(cls_pred == class_l.data).item()
                val_cls_loss = total_loss/len(loader.dataset)
                val_class_acc = class_correct/len(loader.dataset)
            acc = accuracy_score(targets, probas)
            logger.info("Before accuracy: " + str(val_class_acc) + "After accuracy: " + str(acc))
            # print(HEY)
                # breakpoint()
                # acc = accuracy_score(targets,probas)
                # print("Test Accuracy: ", acc)

        else:       # This is the validataion dataloader part
            for it, (data, class_l) in enumerate(loader):
                data, class_l = data.to(self.device), class_l.to(self.device)
                # _, class_logit = self.model(data)
                class_logit = self.model(data)
                # breakpoint()
                _, test_predicted = torch.max(class_logit, 1)
                # breakpoint()
                c = (test_predicted==class_l).squeeze()
                for i in range(data.size(0)):
                    label = class_l[i]
                    class_correct_list[label] += c[i].item()
                    class_total[label] += 1
                criterion = nn.CrossEntropyLoss()
                class_loss = criterion(class_logit, class_l)
                total_loss += class_loss.item()

                _, cls_pred = class_logit.max(dim=1)
                class_correct += torch.sum(cls_pred == class_l.data).item()
                val_cls_loss = total_loss/len(loader.dataset)
                val_class_acc = class_correct/len(loader.dataset)
                # for i in range(3):
                #     if class_total[i] != 0:
                #         # print(phase + " accuracy of " + classes[i] + " is : " + str(class_correct_list[i]/class_total[i]))
                #         logger.info(phase + " accuracy of " + classes[i] + " is : " + str(class_correct_list[i]/class_total[i]))
                # print(total_cls_loss, val_cls_loss)

        # acc = acc()
        # print(class_l, '/n', test_predicted)
        accuracy = accuracy_score(class_l.cpu().detach(), test_predicted.cpu().detach())
        test_predicted = torch.empty(32)
        test_accuracy = str(val_class_acc)
        test_loss = str(val_cls_loss)
        logger.info(str(self.current_epoch) + " " + phase + " " + "SKlearn accuracy: " + str(accuracy)  + " "  + " Cls accuracy: " + test_accuracy + " " + phase + " loss: " + test_loss)
        # if it%10==0:
        #     logger.info(phase + " accuracy: " + test_accuracy + " " + phase + " loss: " + test_loss)
        # breakpoint()
        # for i in range(3):
        #     if class_total[i] != 0:
        #         # print(phase + " accuracy of " + classes[i] + " is : " + str(class_correct_list[i]/class_total[i]))
        #         logger.info(phase + " accuracy of " + classes[i] + " is : " + str(class_correct_list[i]/class_total[i]))

        return class_correct

    def do_training(self, logger):
        # self.logger = Logger(self.args, update_frequency=30)  # , "domain", "lambda"
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            # self.scheduler.step()
            # self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch(logger)
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        string_val_result_max = str(val_res.max())
        string_test_result_max = str(test_res.max())
        string_test_result_p_ind = str(test_res[idx_best])
        print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
        logger.info("Val result max:" + string_val_result_max + "Test resolution" + string_test_result_p_ind + string_test_result_max)
        return self.model


def main():
    args = get_args()
    print("STARTS/.........:)")
    logdir = args.folder_name
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logger = get_logger(os.path.join(logdir, 'trainandval.log'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training(logger)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
