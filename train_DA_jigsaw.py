import argparse

import torch
from IPython.core.debugger import set_trace
from torch import nn
import torch.nn.functional as func
from data import data_helper
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
# from utils.Logger import Logger
import numpy as np
import os
import itertools
import argparse



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

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        model = model_factory.get_network(args.network)(jigsaw_classes=args.jigsaw_n_classes + 1, classes=args.n_classes)
        self.model = model.to(device)
        # print(self.model)
        if args.target in args.source:
            print("No need to include target in source, it is automatically done by this script")
            k = args.source.index(args.target)
            args.source = args.source[:k] + args.source[k + 1:]
            print("Source: %s" % args.source)
        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
        self.target_jig_loader = data_helper.get_target_jigsaw_loader(args)
        self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, target jig: %d, val %d, test %d" % (
            len(self.source_loader.dataset), len(self.target_jig_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all, nesterov=args.nesterov)
        self.jig_weight = args.jig_weight
        self.target_weight = args.target_weight
        self.target_entropy = args.entropy_weight
        self.only_non_scrambled = args.classify_only_sane
        self.n_classes = args.n_classes

    def _do_epoch(self, logger):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for it, (source_batch, target_batch) in enumerate(zip(self.source_loader, itertools.cycle(self.target_jig_loader))):
            (data, jig_l, class_l), d_idx = source_batch
            data, jig_l, class_l, d_idx = data.to(self.device), jig_l.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            tdata, tjig_l, _ = target_batch
            tdata, tjig_l = tdata.to(self.device), tjig_l.to(self.device)

            self.optimizer.zero_grad()

            jigsaw_logit, class_logit = self.model(data)
            jigsaw_loss = criterion(jigsaw_logit, jig_l)
            target_jigsaw_logit, target_class_logit = self.model(tdata)
            target_jigsaw_loss = criterion(target_jigsaw_logit, tjig_l)
            target_entropy_loss = entropy_loss(target_class_logit[tjig_l==0])
            if self.only_non_scrambled:
                class_loss = criterion(class_logit[jig_l == 0], class_l[jig_l == 0])
            else:
                class_loss = criterion(class_logit, class_l)
            _, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)

            loss = class_loss + jigsaw_loss * self.jig_weight + target_jigsaw_loss * self.target_weight + target_entropy_loss * self.target_entropy

            string_jigsaw_loss = str(jigsaw_loss.item())
            string_class_loss  = str(class_loss.item())
            string_jigprediction = str(torch.sum(jig_pred == jig_l.data).item())
            string_cls_prediction = str(torch.sum(cls_pred == class_l.data).item())
            

            if it%10==0:
                logger.info("Epoch no: " + str(self.current_epoch) + "," +  " jigsaw_loss " + string_jigsaw_loss + " class_loss " + string_class_loss + " jigsaw prediction " + string_jigprediction + " class prediction " + string_cls_prediction)

            loss.backward()
            self.optimizer.step()

            # self.logger.log(it, len(self.source_loader),
            #                 {"jigsaw": jigsaw_loss.item(), "class": class_loss.item(), 
            #                  "t_jigsaw": target_jigsaw_loss.item(), "entropy": target_entropy_loss.item()},
            #                 {"jigsaw": torch.sum(jig_pred == jig_l.data).item(),
            #                  "class": torch.sum(cls_pred == class_l.data).item(),
            #                  },
            #                 data.shape[0])
            old_loss = loss
            del loss, class_loss, jigsaw_loss, jigsaw_logit, class_logit, target_jigsaw_logit, target_jigsaw_loss

        
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': old_loss,
            # Add other relevant information if needed
            }, str(self.args.folder_name)+'/epoch'+str(self.current_epoch)+".pth")
        
        del old_loss

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                if loader.dataset.isMulti():
                    jigsaw_correct, class_correct, single_acc = self.do_test_multi(loader)
                    print("Single vs multi: %g %g" % (float(single_acc) / total, float(class_correct) / total))
                else:
                    jigsaw_correct, class_correct = self.do_test(loader, logger, phase)
                jigsaw_acc = float(jigsaw_correct) / total
                class_acc = float(class_correct) / total
                # self.logger.log_test(phase, {"jigsaw": jigsaw_acc, "class": class_acc})
                # self.results[phase][self.current_epoch] = class_acc

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
        
        if phase=="test":       # This is the test dataloader part
            for it, ((data, jig_l, class_l), _) in enumerate(loader):
                data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)
                jigsaw_logit, class_logit = self.model(data)
                _, test_predicted = torch.max(class_logit, 1)
                c = (test_predicted==class_l).squeeze()
                for i in range(data.size(0)):
                    label = class_l[i]
                    class_correct_list[label] += c[i].item()
                    class_total[label] += 1
                criterion = nn.CrossEntropyLoss()
                class_loss = criterion(class_logit, class_l)
                total_loss += class_loss.item()

                _, cls_pred = class_logit.max(dim=1)
                _, jig_pred = jigsaw_logit.max(dim=1)
                class_correct += torch.sum(cls_pred == class_l.data).item()
                jigsaw_correct += torch.sum(jig_pred == jig_l.data).item()
                val_cls_loss = total_loss/len(loader.dataset)
                val_class_acc = class_correct/len(loader.dataset)
                # for i in range(3):
                #     if class_total[i] != 0:
                #         # print(phase + " accuracy of " + classes[i] + " is : " + str(class_correct_list[i]/class_total[i]))
                #         logger.info(phase + " accuracy of " + classes[i] + " is : " + str(class_correct_list[i]/class_total[i]))
                # print(total_cls_loss, val_cls_loss)
                # if it%10==0:
                #     logger.info("Test accuracy: " + str(total_cls_acc/count) + " Test loss: " + str(total_cls_loss/count), " " +  " Iterations count " + str(count))


        else:       # This is the validataion dataloader part
            for it, ((data, jig_l, class_l), _) in enumerate(loader):
                data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)
                jigsaw_logit, class_logit = self.model(data)
                # breakpoint()
                _, val_predicted = torch.max(class_logit, 1)
                c = (val_predicted==class_l).squeeze()
                c_car = val_predicted==1
                for i in range(data.size(0)):
                    label = class_l[i]
                    class_correct_list[label] += c[i].item()
                    class_total[label] += 1
                criterion = nn.CrossEntropyLoss()
                class_loss = criterion(class_logit, class_l)
                total_loss += class_loss.item()

                _, cls_pred = class_logit.max(dim=1)
                _, jig_pred = jigsaw_logit.max(dim=1)
                class_correct += torch.sum(cls_pred == class_l.data).item()
                jigsaw_correct += torch.sum(jig_pred == jig_l.data).item()
                val_cls_loss = total_loss/len(loader.dataset)
                val_class_acc = class_correct/len(loader.dataset)
                string_val_class_acc = str(class_correct/len(loader.dataset))
                # for i in range(3):
                #     if class_total[i] != 0:
                #         # print(phase + " accuracy of " + classes[i] + " is : " + str(class_correct_list[i]/class_total[i]))
                #         logger.info(phase + " accuracy of " + classes[i] + " is : " + str(class_correct_list[i]/class_total[i]))
                # print(total_cls_loss, val_cls_loss)
        test_accuracy = str(val_class_acc)
        test_loss = str(val_cls_loss)
        logger.info(str(self.current_epoch) + " " + phase + " accuracy: " + test_accuracy + " " + phase + " loss: " + test_loss)
        # if it%10==0:
        #     logger.info(phase + " accuracy: " + test_accuracy + " " + phase + " loss: " + test_loss)
        # breakpoint()
        for i in range(3):
            if class_total[i] != 0:
                # print(phase + " accuracy of " + classes[i] + " is : " + str(class_correct_list[i]/class_total[i]))
                logger.info(phase + " accuracy of " + classes[i] + " is : " + str(class_correct_list[i]/class_total[i]))

        return jigsaw_correct, class_correct

    def do_training(self, logger):
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            self._do_epoch(logger)
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        string_val_result_max = str(val_res.max())
        string_test_result_max = str(test_res.max())
        string_test_result_p_ind = str(test_res[idx_best])

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
