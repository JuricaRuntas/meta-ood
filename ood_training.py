import argparse
import time
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from config_helpers import config_training_setup
from src.imageaugmentations import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from src.model_utils import load_network
from torch.utils.data import DataLoader

try:
    from src.metaseg.metrics import mark_segment_borders
except ImportError:
    mark_segment_borders = None
    print("MetaSeg ImportError: Maybe need to compile (src/metaseg/)metrics.pyx ....")
    exit()


def cross_entropy(logits, targets, enc_out_bd, enc_out_in):
    """
    cross entropy loss with one/all hot encoded targets -> logits.size()=targets.size()
    :param logits: torch tensor with logits obtained from network forward pass
    :param targets: torch tensor one/all hot encoded
    :return: computed loss
    """
    neg_log_like = - 1.0 * F.log_softmax(logits, 1)

    if enc_out_bd is None:
        L = torch.mul(targets.float(), neg_log_like)
        print(f"Unweighted in distribution loss: {(L/0.1).mean().item()}")
        L = L.mean()
        return L
    
    L1 = torch.mul(enc_out_bd.float().cuda(), neg_log_like)
    L2 = torch.mul(enc_out_in.float().cuda(), neg_log_like)

    return L1.mean()+L2.mean()


def encode_target(target, pareto_alpha, num_classes, ignore_train_ind, ood_ind=254):
    """
    encode target tensor with all hot encoding for OoD samples
    :param target: torch tensor
    :param pareto_alpha: OoD loss weight
    :param num_classes: number of classes in original task
    :param ignore_train_ind: void class in original task
    :param ood_ind: class label corresponding to OoD class
    :return: one/all hot encoded torch tensor
    """
    npy = target.numpy()
    npz = npy.copy()
    border_ind = min(0, *ignore_train_ind, ood_ind)-1
    borders = np.zeros(npy.shape)

    if np.sum(np.isin(npy, ood_ind)) > 0:
        for i in range(npy.shape[0]):
            borders[i] = mark_segment_borders(npy[i,:,:].astype(np.uint8))

    npy[np.isin(npy, ood_ind)] = num_classes
    npy[np.isin(npy, ignore_train_ind)] = num_classes + 1
    enc = np.eye(num_classes + 2)[npy][..., :-2]  # one hot encoding with last 2 axis cutoff
    
    enc_out_bd = None

    if np.sum(np.isin(npy, num_classes)) > 0: # only if we are looking at the OoD training example
        npc = npy.copy()

        for i in range(npy.shape[0]):
          npc[i][borders[i] < 0] = border_ind

        enc_out_bd = enc.copy()

        enc_out_bd[(npc == border_ind)] = np.full(num_classes, 0.2 / num_classes)
        enc_out_bd[(npc != border_ind)] = np.zeros(num_classes)
        enc_out_bd = torch.from_numpy(enc_out_bd)
        enc_out_bd = enc_out_bd.permute(0, 3, 1, 2).contiguous()

        enc[(np.logical_and(npc != border_ind, npc == num_classes))] = np.full(num_classes, 0.7 / num_classes)
        enc[(npc == border_ind)] = np.zeros(num_classes)
    else:
        enc[(enc == 1)] = 1 - pareto_alpha  # convex combination between in and out distribution samples
    
    enc[np.isin(npz, ignore_train_ind)] = np.zeros(num_classes)
    enc = torch.from_numpy(enc)
    enc = enc.permute(0, 3, 1, 2).contiguous()
    return enc, enc_out_bd, enc


def training_routine(config):
    """Start OoD Training"""
    print("START OOD TRAINING")
    params = config.params
    roots = config.roots
    dataset = config.dataset()
    print("Pareto alpha:", params.pareto_alpha)
    start_epoch = params.training_starting_epoch
    epochs = params.num_training_epochs
    start = time.time()

    """Initialize model"""
    if start_epoch == 0:
        network = load_network(model_name=roots.model_name, num_classes=dataset.num_classes,
                               ckpt_path=roots.init_ckpt, train=True)
    else:
        basename = roots.model_name + "_epoch_" + str(start_epoch) \
                   + "_alpha_" + str(params.pareto_alpha) + ".pth"
        network = load_network(model_name=roots.model_name, num_classes=dataset.num_classes,
                               ckpt_path=os.path.join(roots.weights_dir, basename), train=True)

    transform = Compose([RandomHorizontalFlip(), RandomCrop(params.crop_size), ToTensor(),
                         Normalize(dataset.mean, dataset.std)])

    for epoch in range(start_epoch, start_epoch + epochs):
        """Perform one epoch of training"""
        print('\nEpoch {}/{}'.format(epoch + 1, start_epoch + epochs))
        optimizer = optim.Adam(network.parameters(), lr=params.learning_rate)
        trainloader = config.dataset('train', transform, roots.cs_root, roots.coco_root, params.ood_subsampling_factor)
        dataloader = DataLoader(trainloader, batch_size=params.batch_size, shuffle=True)
        i = 0
        loss = None
        for x, target in dataloader:
            optimizer.zero_grad()
            logits = network(x.cuda())
            y, enc_out_bd, enc_out_in = encode_target(target=target, pareto_alpha=params.pareto_alpha, 
                                                      num_classes=dataset.num_classes, ignore_train_ind=dataset.void_ind, 
                                                      ood_ind=dataset.train_id_out)
            loss = cross_entropy(logits, y.cuda(), enc_out_bd, enc_out_in)
            loss.backward()
            optimizer.step()
            print('{} Loss: {}'.format(i, loss.item()))
            i += 1

        """Save model state"""
        save_basename = roots.model_name + "_epoch_" + str(epoch + 1) + "_alpha_" + str(params.pareto_alpha) + ".pth"
        print('Saving checkpoint', os.path.join(roots.weights_dir, save_basename))
        torch.save({
            'epoch': epoch + 1,
            'state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(roots.weights_dir, save_basename))
        torch.cuda.empty_cache()

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("FINISHED {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def main(args):
    """Perform training"""
    config = config_training_setup(args)
    training_routine(config)


if __name__ == '__main__':
    """Get Arguments and setup config class"""
    parser = argparse.ArgumentParser(description='OPTIONAL argument setting, see also config.py')
    parser.add_argument("-train", "--TRAINSET", nargs="?", type=str)
    parser.add_argument("-model", "--MODEL", nargs="?", type=str)
    parser.add_argument("-epoch", "--training_starting_epoch", nargs="?", type=int)
    parser.add_argument("-nepochs", "--num_training_epochs", nargs="?", type=int)
    parser.add_argument("-alpha", "--pareto_alpha", nargs="?", type=float)
    parser.add_argument("-lr", "--learning_rate", nargs="?", type=float)
    parser.add_argument("-crop", "--crop_size", nargs="?", type=int)
    main(vars(parser.parse_args()))
