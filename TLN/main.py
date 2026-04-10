# -*- coding: utf-8 -*-
import os, time, sys, argparse
from pprint import pprint

import numpy as np
import random

import torch, PIL
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms

import models
from utils import my_dataset, Logger, load_pickle,plot_anomaly_score_distribution
from NDCC import NDCC
from classifier import Classifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='classifier', choices=['NDCC', 'classifier'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test', action='store_true', default=False, help='Test mode.')
    parser.add_argument('--data_folder', type=str,
                        default='/xiliang/GMH/process_data/ND_data')
    parser.add_argument('--dataset', type=str, default='HAR_inertial',
                        choices=['WISDM', 'GRABMyo', 'DailySports', 'UWave', 'HAR_inertial'])
    parser.add_argument('--network', type=str, default='cnn',
                        choices=['cnn', 'transformer', 'alexnet', 'vgg16'])
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=42, help='random seed for train/test split')
    parser.add_argument('--strategy', type=int, default=3, choices=[1, 2],
                        help=r'strategy for the parameterization of $\Sigma$')

    parser.add_argument('--num_classes', type=int, default=100, help='the number of training classes')
    parser.add_argument('--num_epochs', type=int, default=10, help='the number of training epochs')

    parser.add_argument('--lr1', type=float, default=1e-3, help='learning rate for embedding v(x)')
    parser.add_argument('--lr2', type=float, default=1e-1, help='learning rate for linear classifier {w_y, b_y}')
    parser.add_argument('--lr3', type=float, default=1e-1, help=r'learning rate for \sigma')
    parser.add_argument('--lr4', type=float, default=1e-3, help=r'learning rate for \delta_j')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr_milestones', default=[5, 10])

    parser.add_argument('--lmd', type=float, default=2e-1, help=r'\lambda in Eq. (23)')
    parser.add_argument('--gma', type=float, default=1 / 4096, help=r'\gamma in Eq. (22)')
    parser.add_argument('--r', type=float, default=16, help=r'\|v(x)\|=r')
    parser.add_argument('--d', type=int, default=4096, help='dimentionality of v(x)')

    parser.add_argument('--exp_id', type=str, default='1')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device set to train.')

    opt = parser.parse_args()

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # Python 内置随机种子
    random.seed(opt.seed)

    # Numpy 随机种子
    np.random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # import socket
    # opt.exp_id = socket.gethostname()

    output_folder = os.path.join(opt.checkpoint_dir, opt.dataset, opt.network, opt.exp_id)
    os.makedirs(output_folder, exist_ok=True)

    log_file = os.path.join(output_folder, 'log.out')
    err_file = os.path.join(output_folder, 'err.out')

    sys.stdout = Logger(log_file)
    sys.stderr = Logger(err_file)  # redirect std err, if necessary

    # recommended choice for hyperparameters (according to Table C.1. in our Supplementary Material)
    if opt.dataset == 'StanfordDogs':
        opt.num_classes = 60
        opt.lr1 = 1e-3
        opt.lr2 = 1e-1
        opt.lr3 = 1e-1
        opt.lr4 = 1e-3
        opt.r = 16
        opt.lmd = 2e-1

        opt.lr_milestones = [25, 28, 30]
        opt.num_epochs = 30

    elif opt.dataset == 'WISDM':
        opt.num_classes = 9
        opt.lr1 = 1e-2
        opt.lr2 = 1e-2
        opt.lr3 = 1e-2
        opt.lr4 = 1e-2
        opt.r = 16
        opt.d = 256
        opt.lmd = 2e-1
        opt.lr_milestones = [5, 10]
        opt.num_epochs = 1000
    elif opt.dataset == 'DailySports':
        opt.num_classes = 9
        opt.lr1 = 1e-2
        opt.lr2 = 1e-2
        opt.lr3 = 1e-2
        opt.lr4 = 1e-2
        opt.r = 16
        opt.d = 256
        opt.lmd = 2e-1
        opt.lr_milestones = [5, 10]
        opt.num_epochs = 200
    elif opt.dataset == 'GRABMyo':
        opt.num_classes = 8
        opt.lr1 = 1e-2
        opt.lr2 = 1e-2
        opt.lr3 = 1e-2
        opt.lr4 = 1e-2
        opt.r = 16
        opt.d = 256
        opt.lmd = 2e-1
        opt.lr_milestones = [5, 10]
        opt.num_epochs = 400
    elif opt.dataset == 'UWave':
        opt.num_classes = 4
        opt.lr1 = 1e-2
        opt.lr2 = 1e-2
        opt.lr3 = 1e-2
        opt.lr4 = 1e-2
        opt.r = 16
        opt.d = 256
        opt.lmd = 2e-1
        opt.lr_milestones = [5, 10]
        opt.num_epochs = 300
    elif opt.dataset == 'HAR_inertial':
        opt.num_classes = 3
        opt.lr1 = 1e-2
        opt.lr2 = 1e-2
        opt.lr3 = 1e-2
        opt.lr4 = 1e-2
        opt.r = 16
        opt.d = 256
        opt.lmd = 2e-1
        opt.lr_milestones = [5, 10]
        opt.num_epochs = 100

    pprint(vars(opt))

    # images, labels = pd.read_csv(opt.dataset + '.csv', sep=',').values.transpose()

    path = opt.data_folder + '/' + opt.dataset + '/'
    x_train = load_pickle(path + 'x_train.pkl')
    y_train = load_pickle(path + 'state_train.pkl')
    x_test = load_pickle(path + 'x_test.pkl')
    y_test = load_pickle(path + 'state_test.pkl')

    # known/unknown split
    known_train_idx = np.where(y_train < opt.num_classes)[0]
    unknown_train_idx = np.where(y_train >= opt.num_classes)[0]
    known_test_idx = np.where(y_test < opt.num_classes)[0]
    unknown_test_idx = np.where(y_test >= opt.num_classes)[0]

    known_train = x_train[known_train_idx]
    known_train_labels = y_train[known_train_idx]
    known_test = x_test[known_test_idx]
    known_test_labels = y_test[known_test_idx]

    train_dataset = my_dataset(known_train, known_train_labels)
    val_dataset = my_dataset(known_test, known_test_labels)

    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True)

    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = val_loader

    dataset_sizes = {}
    dataset_sizes['train'] = len(train_dataset)
    dataset_sizes['val'] = len(val_dataset)
    C = x_train.shape[-1]

    if opt.network == 'vgg16':
        embedding = models.vgg16(pretrained=True)
        embedding.classifier[6] = nn.Sequential()
    elif opt.network == 'alexnet':
        embedding = models.alexnet(pretrained=True)
        embedding.classifier[6] = nn.Sequential()
    elif opt.network == 'transformer':
        embedding = models.transformer(input_dim=C,
                                       embed_dim=opt.d,
                                       depth=1,
                                       num_heads=4,
                                       num_classes=opt.num_classes,
                                       dropout=0.2)
        embedding.classifier[2] = nn.Sequential()
    elif opt.network == 'cnn':
        embedding = models.cnn(in_channels=C, num_classes=opt.num_classes)
        embedding.classifier[0] = nn.Sequential()

    classifier = nn.Linear(opt.d, opt.num_classes)
    if opt.model == "NDCC":
        model = NDCC(embedding=embedding, classifier=classifier, opt=opt, l2_normalize=True)
    elif opt.model == "classifier":
        model = Classifier(embedding=embedding, classifier=classifier, opt=opt)

    if opt.strategy == 1:
        optimizer = torch.optim.SGD([{'params': model.embedding.parameters(), 'lr': opt.lr1},
                                     {'params': model.classifier.parameters(), 'lr': opt.lr2, 'weight_decay': 0e-4},
                                     {'params': [model.sigma], 'lr': opt.lr3, 'weight_decay': 0e-4},
                                     ], momentum=opt.momentum, weight_decay=5e-4)
    elif opt.strategy == 2:
        optimizer = torch.optim.SGD([{'params': model.embedding.parameters(), 'lr': opt.lr1},
                                     {'params': model.classifier.parameters(), 'lr': opt.lr2, 'weight_decay': 0e-4},
                                     {'params': [model.sigma], 'lr': opt.lr3, 'weight_decay': 0e-4},
                                     {'params': [model.delta], 'lr': opt.lr4, 'weight_decay': 0e-4},
                                     ], momentum=opt.momentum, weight_decay=5e-4)
    else:
        optimizer = torch.optim.SGD([{'params': model.embedding.parameters(), 'lr': opt.lr1},
                                     {'params': model.classifier.parameters(), 'lr': opt.lr2, 'weight_decay': 0e-4},
                                     ], momentum=opt.momentum, weight_decay=5e-4)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_milestones, gamma=0.1)

    assert torch.cuda.is_available()
    model = model.to(opt.device)

    if opt.test == False:
        # ==================== training ====================
        print('training started!')
        model.fit(optimizer=optimizer, scheduler=scheduler, dataloaders=dataloaders,
                  num_epochs=opt.num_epochs)
        print('training finished!')

        saved_model_path = os.path.join(output_folder, 'NDCC_state_dict.pth')
        torch.save(model.state_dict(), saved_model_path)

        # ==================== evaluation ====================

        ND_images = np.vstack([x_test[known_test_idx], x_test[unknown_test_idx], x_train[unknown_train_idx]])

        # binary labels (1 for novel and 0 for seen)
        ND_labels = np.hstack(
            [np.zeros(len(known_test_idx)), np.ones(len(unknown_test_idx)), np.ones(len(unknown_train_idx))])

        ND_dataset = my_dataset(ND_images, ND_labels)
        ND_loader = DataLoader(dataset=ND_dataset, batch_size=opt.test_batch_size, shuffle=False,
                               num_workers=opt.num_workers,
                               pin_memory=True)

        print('evaluation started!')
        model.compute_class_prototypes(train_loader)
        ND_scores = model.get_ND_scores(ND_loader)
        print('AUC ROC: %.4f' % roc_auc_score(ND_labels, ND_scores))

        # PR-AUC
        pr_auc = average_precision_score(ND_labels, ND_scores)
        print('PR-AUC: %.4f' % pr_auc)

        # FPR@95TPR
        fpr, tpr, th = roc_curve(ND_labels, ND_scores)
        fpr95 = fpr[np.where(tpr >= 0.95)[0][0]]
        print('FPR@95TPR: %.4f' % fpr95)
        print('evaluation finished!')
    else:
        # ==================== evaluation ====================
        saved_model_path = os.path.join(output_folder, 'NDCC_state_dict.pth')
        state_dict = torch.load(saved_model_path, map_location=opt.device)
        model.load_state_dict(state_dict)

        ND_images = np.vstack([x_test[known_test_idx], x_test[unknown_test_idx], x_train[unknown_train_idx]])

        # binary labels (1 for novel and 0 for seen)
        ND_labels = np.hstack(
            [np.zeros(len(known_test_idx)), np.ones(len(unknown_test_idx)), np.ones(len(unknown_train_idx))])

        ND_dataset = my_dataset(ND_images, ND_labels)
        ND_loader = DataLoader(dataset=ND_dataset, batch_size=opt.test_batch_size, shuffle=False,
                               num_workers=opt.num_workers,
                               pin_memory=True)

        print('evaluation started!')

        model.compute_class_prototypes(train_loader)
        ND_scores = model.get_ND_scores(ND_loader)

        print('AUC ROC: %.4f' % roc_auc_score(ND_labels, ND_scores))

        # PR-AUC
        pr_auc = average_precision_score(ND_labels, ND_scores)
        print('PR-AUC: %.4f' % pr_auc)

        # FPR@95TPR
        fpr, tpr, th = roc_curve(ND_labels, ND_scores)
        fpr95 = fpr[np.where(tpr >= 0.95)[0][0]]

        print('FPR@95TPR: %.4f' % fpr95)
        # 计算阈值（95%分位数）
        ood_ratio = np.mean(ND_labels == 1)  # π
        threshold = np.percentile(ND_scores, 100 * (1 - ood_ratio))

        # 二分类

        ND_pred = (ND_scores >= threshold).astype(int)

        plot_anomaly_score_distribution(ND_scores, ND_labels, ND_pred,
                                        plot_adjusted=False, save_dir=f"./score_view/{opt.dataset}/",
                                        error_margin=0.2)
        print('evaluation finished!')
