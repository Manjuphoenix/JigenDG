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
import tqdm
from skimage.io import imsave
import cv2



from torchvision import models


# device_ids = [2,3]
# devices = [torch.device(f'cuda:{id}') for id in device_ids]


devices = torch.device('cuda' if torch.cuda.is_available else 'cpu' )

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
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=31, help="Number of classes")
    parser.add_argument("--jigsaw_n_classes", "-jc", type=int, default=31, help="Number of classes for the jigsaw task")
    parser.add_argument("--source_network", choices=model_factory.nets_map.keys(), help="Which network to use", default="caffenet")
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


# logdir = "outputs/jigen"



# if not os.path.exists(args.folder_name):
#     os.makedirs(logdir)
# logger = get_logger(os.path.join(logdir, 'trainandval.log'))

# Gradcam
class SaveGradient:
    def __init__(self):
        self.gradient = None
    def __call__(self, grad):
        self.gradient = grad
        
hook = SaveGradient()


# Jigen transforms for target only...............
def get_train_transformers(args):
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

def get_target_jigsaw_loader(args):
    img_transformer, tile_transformer = get_train_transformers(args)
    name_train, _, labels_train, _ = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % args.target), 0)
    dataset = JigsawDataset(name_train, labels_train, patches=False, img_transformer=img_transformer,
                            tile_transformer=tile_transformer, jig_classes=args.jigsaw_n_classes, bias_whole_image=args.bias_whole_image)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    return loader





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



class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return func.normalize(self.proj(x), p=2, dim=1)
    

projection_head = ProjectionHead(dim_in=2048, proj="linear").to("cuda")

# Contrastive loss:
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, eps=1e-8):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z_i, z_j):
        # Reshape embeddings to 2D tensors
        z_i_flat = z_i.view(z_i.size(0), -1)
        z_j_flat = z_j.view(z_j.size(0), -1)
        
        # Concatenate the feature vectors
        z = torch.cat([z_i_flat, z_j_flat], dim=0)
        
        # Compute pairwise cosine similarity
        sim_matrix = torch.matmul(z, z.t()) / self.temperature
        
        # Normalize values
        sim_matrix = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]
        
        # Fill the diagonal with high negative values to prevent same-sample comparison
        mask = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -1e9)
        
        # Compute numerator (log-sum-exp)
        numerator = torch.exp(sim_matrix).sum(dim=1)
        
        # Compute denominator (log-sum-exp) for all samples
        denominator = torch.exp(sim_matrix).sum(dim=1, keepdim=True)
        
        # Avoid division by zero
        denominator = torch.clamp(denominator, min=self.eps)
        
        # Compute loss
        loss = -torch.log(numerator / denominator)
        
        # Take the mean across batch
        loss = loss.mean()
        # print(loss)
        # breakpoint()
        return loss

    
contrastive_loss = SupConLoss()


# Define hook to extract gradients from the target layer
class SaveGradient:
    def __init__(self):
        self.gradient = None
    def __call__(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.source_model =  model_factory.get_network(args.network)(jigsaw_classes=args.jigsaw_n_classes + 1, classes=args.n_classes)
        # self.source_model = nn.DataParallel(self.source_model, device_ids=device_ids).to(devices[0])
        self.target_model = model_factory.get_network(args.network)(jigsaw_classes=args.jigsaw_n_classes + 1, classes=args.n_classes)
        # self.target_model = nn.DataParallel(self.target_model, device_ids=device_ids).to(devices[0])
        # print(self.model)
        if args.target in args.source:
            print("No need to include target in source, it is automatically done by this script")
            k = args.source.index(args.target)
            args.source = args.source[:k] + args.source[k + 1:]
            print("Source: %s" % args.source)
        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=self.source_model.is_patch_based())
        self.target_jig_loader = data_helper.get_target_jigsaw_loader(args)
        self.target_loader = data_helper.get_val_dataloader(args, patches=self.source_model.is_patch_based())
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, target jig: %d, val %d, test %d" % (
            len(self.source_loader.dataset), len(self.target_jig_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(self.source_model, args.epochs, args.learning_rate, args.train_all, nesterov=args.nesterov)
        self.jig_weight = args.jig_weight
        self.target_weight = args.target_weight
        self.target_entropy = args.entropy_weight
        self.only_non_scrambled = args.classify_only_sane
        self.n_classes = args.n_classes
        
        

    def _do_epoch(self, logger):
        criterion = nn.CrossEntropyLoss()
        self.source_model.train()
        self.source_model.to("cuda")
        # Register hook
        src_hook = SaveGradient()
        tgt_hook = SaveGradient()
        src_img = cv2.imread("420.jpg")
        tgt_img = cv2.imread("FLIR_135.jpeg")
        # it is Iterations count with source and target batch................
        for it, (source_batch, target_batch) in enumerate(zip(self.source_loader, itertools.cycle(self.target_jig_loader))):
            # Source loader.............
            (data, jig_l, class_l), d_idx = source_batch
            # class_l is the classification label y, data is the batch of tensors (images), jig_l is the permutation index, 
            data, jig_l, class_l, d_idx = data.to(self.device), jig_l.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            # cv2.imwrite("data1.png", np.moveaxis(data[0].detach().cpu().numpy(), 0, -1))
            # breakpoint()
            # imsave("data0.png", np.moveaxis(data[0].detach().cpu().numpy(), 0, -1))
            # print(HEY)
            # breakpoint()
            # Target loader.............
            tdata, tjig_l, _ = target_batch
            tdata, tjig_l = tdata.to(self.device), tjig_l.to(self.device)

            hook_src_handle = self.source_model.layer4.register_backward_hook(hook)
            hook_tgt_handle = self.source_model.layer4.register_backward_hook(hook)
            
            
            self.optimizer.zero_grad()

            # This is the source image and label passed to the common feature extractor
            jigsaw_logit, class_logit, x4_source = self.source_model(data)
            
            # For source model
            predicted_src_class = torch.argmax(class_logit, dim=1)
            
            # hook_src_handle = x4_source.register_backward_hook(hook)
            x4_src_prj = projection_head(x4_source)
            
            
            x4_source = x4_src_prj.unsqueeze(dim=1)
            # jig_l is the class label for each patch and jigsaw_logit is the predicted class label and jigsaw_loss is the classification loss
            jigsaw_loss = criterion(jigsaw_logit, jig_l)
            
            # This is the target image and label passed to the common feature extractor
            target_jigsaw_logit, target_class_logit, x4_target = self.source_model(tdata)
            
            # GradCAM for target model
            predicted_tgt_class = torch.argmax(target_class_logit, dim=1)
            if torch.is_tensor(predicted_class):
                predicted_src_class = predicted_src_class.item()
                predicted_tgt_class = predicted_tgt_class.item()
            
            # print(class_logit, class_logit.shape)
            # print(target_class_logit, target_class_logit.shape)
            # print(HEY)
            # Backward pass to compute gradients
            self.source_model.zero_grad()
            self.target_model.zero_grad()
            class_logit[0, predicted_src_class].backward()
            target_class_logit[0, predicted_tgt_class].backward()
            
            # Compute importance weights using gradients
            src_grads = src_hook.gradient
            tgt_grads = tgt_hook.gradient
            
            src_pooled_grads = torch.mean(src_grads, dim=(2, 3), keepdim=True)
            tgt_pooled_grads = torch.mean(tgt_grads, dim=(2, 3), keepdim=True)
            
            src_activations = self.source_model.layer4[-1].conv3.forward(data)
            tgt_activations = self.target_model.layer4[-1].conv3.forward(tdata)
            
            src_weighted_activations = src_activations * src_pooled_grads
            tgt_weighted_activations = tgt_activations * tgt_pooled_grads
            
            src_importance_weights = torch.mean(src_weighted_activations, dim=1, keepdim=True)
            tgt_importance_weights = torch.mean(tgt_weighted_activations, dim=1, keepdim=True)
            
            
            # Compute heat map
            src_heat_map = src_importance_weights.relu().squeeze().detach().numpy()
            tgt_heat_map = tgt_importance_weights.relu().squeeze().detach().numpy()
            
            # Upsample heat map to input image size for source
            src_heat_map = np.maximum(0, src_heat_map)
            src_heat_map = src_heat_map - np.min(src_heat_map)
            src_heat_map = src_heat_map / np.max(src_heat_map)
            src_heat_map = np.uint8(255 * src_heat_map)
            src_heat_map = np.uint8(255 * src_heat_map / np.max(src_heat_map))
            
            # Upsample heat map to input image size for target
            tgt_heat_map = np.maximum(0, tgt_heat_map)
            tgt_heat_map = tgt_heat_map - np.min(tgt_heat_map)
            tgt_heat_map = tgt_heat_map / np.max(tgt_heat_map)
            tgt_heat_map = np.uint8(255 * tgt_heat_map)
            tgt_heat_map = np.uint8(255 * tgt_heat_map / np.max(tgt_heat_map))
            
            
            
            # Source Visualize heat map overlaid on input image
            plt.imshow(src_img)
            plt.imshow(heat_map, alpha=0.6, cmap='jet')
            plt.axis('off')
            plt.show()

            # Remove hook
            hook_handle.remove()
            
            # Target Visualize heat map overlaid on input image
            plt.imshow(tgt_img)
            plt.imshow(heat_map, alpha=0.6, cmap='jet')
            plt.axis('off')
            plt.show()
            
            print(HEY)

            # Remove hook
            hook_handle.remove()
            
            # hook_tgt_handle = x4_target.register_backward_hook(hook)
            x4_tar_prj = projection_head(x4_target)
            x4_target = x4_tar_prj.unsqueeze(dim=1)
            
            concatinated_feat = torch.cat([x4_source, x4_target], dim=1)
            # Contrastive loss for adaptation:
            contrastive_loss_dothi = contrastive_loss(x4_src_prj, x4_tar_prj)
            # print(contrastive_loss_dothi)
            # breakpoint()
                        
            # Jigsaw loss for target............
            target_jigsaw_loss = criterion(target_jigsaw_logit, tjig_l)
            
            # Softmax loss loss function
            target_entropy_loss = entropy_loss(target_class_logit[tjig_l==0])
            # Not scrambled is not shuffling i.e. here we do only classification of the source image...........
            # if self.only_non_scrambled and self:      #old code
            if self.only_non_scrambled:        # update for our code run the F2 encoder only for 2 epochs......
                class_loss = criterion(class_logit[jig_l == 0], class_l[jig_l == 0])
            # Here the target image will be scrambled hence compute the permutation index for each of the logits or for each patch of the image.......
            else:
                class_loss = criterion(class_logit, class_l)
            _, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)
            # breakpoint()
            loss = class_loss + jigsaw_loss * self.jig_weight + target_jigsaw_loss * self.target_weight + target_entropy_loss * self.target_entropy

            loss.backward()
            self.optimizer.step()

            logger.info(" jigsaw_loss " + str(jigsaw_loss.item()) + " class_loss " + str(class_loss.item()) + " jigsaw prediction " + str(torch.sum(jig_pred == jig_l.data).item()) + " class prediction " + str(torch.sum(cls_pred == class_l.data).item()))
            
            # self.logger.log(it, len(self.source_loader),
            #                 {"jigsaw": jigsaw_loss.item(), "class": class_loss.item(), 
            #                  "t_jigsaw": target_jigsaw_loss.item(), "entropy": target_entropy_loss.item()},
            #                 {"jigsaw": torch.sum(jig_pred == jig_l.data).item(),
            #                  "class": torch.sum(cls_pred == class_l.data).item(),
            #                  },
            #                 data.shape[0])
            old_loss = loss
            del loss, class_loss, jigsaw_loss, jigsaw_logit, class_logit, target_jigsaw_logit, target_jigsaw_loss
        
        # For inferencing and calculating the results.................
        # checkpoint = torch.load('./outputs/jigen-da-run4-lr-changed-to-polylr/epoch6.pth')
        # self.source_model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.source_model.eval()
        # torch.save({
        #     'epoch': self.current_epoch,
        #     'model_state_dict': self.model.state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     'loss': old_loss,
        #     # Add other relevant information if needed
        #     }, './outputs/'+str(self.args.folder_name)+'/epoch'+str(self.current_epoch)+".pth")
        # del old_loss
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                print(phase, "----------------")
                total = len(loader.dataset)
                if loader.dataset.isMulti():
                    print("Multi..................")
                    jigsaw_correct, class_correct, single_acc = self.do_test_multi(loader)
                    print("Single vs multi: %g %g" % (float(single_acc) / total, float(class_correct) / total))
                else:
                    jigsaw_correct, class_correct = self.do_test(loader)
                jigsaw_acc = float(jigsaw_correct) / total
                class_acc = float(class_correct) / total
                string_jigsaw_acc = str(jigsaw_acc)
                string_class_acc = str(class_acc)
                # logger.info(phase, {"jigsaw": jigsaw_acc, "class": class_acc})
                # breakpoint()
                print(phase, " jigsaw ", jigsaw_acc, " class ", class_acc)
                logger.info(phase + " jigsaw " + string_jigsaw_acc + " class " + string_class_acc)
                # self.logger.log_test(phase, {"jigsaw": jigsaw_acc, "class": class_acc})
                self.results[phase][self.current_epoch] = class_acc
                # print("Val Results", self.results[phase][self.current_epoch])

    def do_test(self, loader):
        jigsaw_correct = 0
        class_correct = 0
        domain_correct = 0
        # print(self.current_epoch)
        self.source_model = self.source_model.to(devices)
        for it, ((data, jig_l, class_l), _) in enumerate(loader):
            data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)
            jigsaw_logit, class_logit, x4_layer = self.source_model(data)
            _, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
            jigsaw_correct += torch.sum(jig_pred == jig_l.data)
        # save_model = self.model()
        # torch.save(save_model.state_dict(), "./outputs/source_only_clean_code_run2/epoch"+str(self.current_epoch)+".pth")
        return jigsaw_correct, class_correct

    def do_training(self, logger):
        # self.logger = Logger(self.args, update_frequency=30)  # , "domain", "lambda"
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            # self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch(logger)
            
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        logger.info("Val result max:" + str(val_res.max()) + "Test resolution" + str(test_res[idx_best]) + str(test_res.max()))
        print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
        # self.logger.save_best(test_res[idx_best], test_res.max())
        return  self.model


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
