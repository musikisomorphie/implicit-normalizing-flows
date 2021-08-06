import logging
from Data import SCRC_dict
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler, RandomSampler
import matplotlib.pyplot as plt
import time
import os
import copy
from pathlib import Path
from random import shuffle
# from VISION.utils import custom_logger, initialize_model, get_data_loader, get_transform, SCRC
import sys
sys.path.append('.')


def compute_acc(grads, grad_labs, phase):
    """Compute the classification accuracies achieved by trained model 

    Args:
        grads: the predicted grades
        grad_labs: the ground-truth grades
        phase: train, val, test

    Returns:
        all the acc info saved in the logger
        the acc for normal, low grade, and high grade TMA
    """
    acc_tot = torch.sum(grads == grad_labs.data)
    norml_tp = torch.sum((grads == 0) & (grad_labs.data == 0))
    norml_tot = torch.sum(grad_labs.data == 0)

    low_tp = torch.sum((grads == grad_labs.data) & (grad_labs.data == 1))
    low_tot = torch.sum(grad_labs.data == 1)

    high_tp = torch.sum((grads == grad_labs.data) & (grad_labs.data == 2))
    high_tot = torch.sum(grad_labs.data == 2)

    tumor_tp = torch.sum((grads != 0) & (grad_labs.data != 0))
    tumor_fp = torch.sum((grads != 0) & (grad_labs.data == 0))
    tumor_fn = torch.sum((grads == 0) & (grad_labs.data != 0))

    acc_tot = acc_tot.float() / grads.size()[0]

    acc_norml = norml_tp.float() / norml_tot
    acc_low = low_tp.float() / low_tot
    acc_high = high_tp.float() / high_tot

    acc_tumor = (norml_tp.float() + tumor_tp.float()) / grads.size()[0]

    acc_info = ('{} Total acc: {:.2f}%, Normal acc: {:.2f}%, Tumor acc: {:.2f}%, \n'
                '   low acc: {:.2f}%, high acc: {:.2f}% \n'
                '   Tumor Precision: {:.2f}%, Recall: {:.2f}%'). \
        format(phase, acc_tot * 100, acc_norml * 100, acc_tumor * 100,
               acc_low * 100, acc_high * 100,
               tumor_tp * 100 / (tumor_tp + tumor_fp), tumor_tp * 100 / (tumor_tp + tumor_fn))
    return acc_info, acc_norml, acc_low, acc_high


def train_model(batch,
                device,
                model,
                dataloaders,
                criterion,
                optimizer,
                scheduler,
                logging,
                dir_pt,
                num_epochs=25):
    """The main function for training scrc,
    the output is grade classification, 0 Normal TMA, 1 low grade TMA, 2 high grade TMA.

    Args:
        exp: the experiment id 
        batch: batch size for training
        device: CUDA or CPU
        model: the model architecture resnet, densenet, etc.
        dataloader: the dataloader for train, val, test
        criterion: the loss function -- crossentropy
        opitmizer: the optimizer -- Adam
        scheduler: learning rate sceduler, decay LR after certain step size
        logging:  saved logging information with neat format
        dir_pt: path to the output pt file with good accuracy
        num_epochs: the training epochs
    """

    since = time.time()
    best_score = float('-inf')

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ('train', 'val', 'test'):
            dt_num = len(dataloaders[phase].dataset)
            grads_total = torch.zeros(dt_num, device=device)
            grad_labs_total = torch.zeros(dt_num, device=device)
            loss = 0.

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            for idx, dt_scrc in enumerate(dataloaders[phase]):
                imgs = dt_scrc[0].to(device)
                labs = dt_scrc[1].to(device)
                grad_dim = SCRC_dict['map']['G'][0]
                grad_labs = labs[:, grad_dim].long()
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        inputs = imgs
                        grad = model(inputs)
                        loss_grad = criterion(grad, grad_labs)
                        loss_grad.backward()
                        optimizer.step()
                        loss += loss_grad.item() * inputs.size(0)
                    else:
                        inputs = imgs
                        grad_many = model(inputs)
                        grad = grad_many

                    _, class_grad = torch.max(grad, dim=1)
                    grads_total[idx * batch:(idx + 1) * batch] = class_grad
                    grad_labs_total[idx * batch:(idx + 1) * batch] = grad_labs

            acc_info, acc_norml, acc_low, acc_high = compute_acc(
                grads_total, grad_labs_total, phase)
            cur_score = acc_norml + acc_low + acc_high
            cur_cond = acc_norml >= 0.9 and acc_low >= 0.8 and acc_high >= 0.55

            if phase == 'train':
                scheduler.step()
                logging.info('{} CrosEntropyLoss: {:.4f}'.format(
                    phase, loss / dt_num))
            elif phase == 'test':
                acc_info += '\n \n'
            elif phase == 'val' and cur_score > best_score and cur_cond:
                logging.info(
                    'save the model with best score {} at epoch {}'.format(cur_score, epoch))
                best_score = cur_score
                model_file = str(dir_pt) + '_{}.pt'.format(epoch)
                torch.save(model.state_dict(), model_file)

            logging.info(acc_info)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_score))
    return


def main():
    parser = argparse.ArgumentParser(
        description='cnn backbones for Swiss CRC')
    parser.add_argument('--data-dir', type=Path, default='/home/histopath/Data/SCRC/', metavar='DIR',
                        help='input batch size for training')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 8)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 25)')
    parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-results', action='store_true', default=False,
                        help='Save the predicted results for causal inference')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # model configuration
    exp_dir = args.data_dir / 'experiments'
    # 'resnet', 'densenet', 'shufflenet', 'mobilenet'
    for md_name in ('densenet',):
        for id_exp in range(4):
            model = initialize_model(md_name,
                                     num_classes=3,
                                     num_pred=0).to(device)

            # generate the data loader for train, val and test
            data_loader = dict()
            for phase in ('train', 'val', 'test'):
                bat = args.batch_size if phase == 'train' else args.test_batch_size
                dt_trans = get_transform('scrc', phase)
                data_scrc = SCRC(str(args.data_dir /
                                     'gt_{}.pt'.format(phase)),
                                 transform=dt_trans)
                dt_loader = get_data_loader(data_scrc, bat, seq=False)
                data_loader[phase] = dt_loader
                print('{} data_loader done.'.format(phase))

            criterion = nn.CrossEntropyLoss()

            optimizer_ft = optim.AdamW(model.parameters(), lr=args.lr)

            # Decay LR by a factor of 0.1 every 20 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(
                optimizer_ft, gamma=args.gamma, step_size=20)

            single_exp_dir = exp_dir / ('cnn_{}'.format(id_exp))
            logger = custom_logger(
                str(single_exp_dir / ('{}.log'.format(md_name))))

            dir_pt = args.data_dir / 'experiments' / \
                'cnn_{}'.format(id_exp) / '{}'.format(md_name)

            train_model(args.batch_size,
                        device,
                        model,
                        data_loader,
                        criterion,
                        optimizer_ft,
                        exp_lr_scheduler,
                        logger,
                        dir_pt,
                        num_epochs=args.epochs)


if __name__ == '__main__':
    main()
