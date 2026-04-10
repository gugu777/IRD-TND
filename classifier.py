# -*- coding: utf-8 -*-
import os, time, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import mahalanobis_metric
from tqdm import tqdm




class Classifier(nn.Module):
    def __init__(self, embedding, classifier, opt):
        super(Classifier, self).__init__()
        self.embedding = embedding
        self.classifier = classifier

        self.dim_embedding = self.classifier.in_features
        self.num_classes = self.classifier.out_features

        self.strategy = opt.strategy
        self.lmd = opt.lmd
        self.r = opt.r
        self.gma = opt.gma
        self.device = opt.device
        # if self.strategy == 1:
        #     self.sigma = torch.tensor(((np.ones(1))).astype('float32'), requires_grad=True, device=self.device)
        #     self.delta = None
        #
        # elif self.strategy == 2:
        #     self.sigma = torch.tensor(((np.ones(1))).astype('float32'), requires_grad=True, device=self.device)
        #     self.delta = torch.tensor((np.zeros(self.dim_embedding)).astype('float32'), requires_grad=True,
        #                               device=self.device)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.r * torch.nn.functional.normalize(x, p=2, dim=1)

        return x

    def fit(self, optimizer, scheduler, dataloaders, num_epochs=20):
        for epoch in range(num_epochs):
            since2 = time.time()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            # for phase in ['train', 'val']:

            for phase in ['train']:

                self.eval()  # NDCC is alwayes set to evaluate mode

                cnt = 0

                epoch_loss = 0.
                epoch_acc = 0.

                for step, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):

                    inputs = (inputs.to(self.device))
                    labels = (labels.long().to(self.device))

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self(inputs)
                        logits = self.classifier(outputs)
                        loss_CE = F.cross_entropy(logits, labels)
                        loss = loss_CE

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # if step % 100 == 0:
                    #     print('{} step: {} loss: {:.4f}, loss_CE: {:.4f}, loss_MD: {:.4f}, loss_NLL: {:.4f}'.format(
                    #         phase, step, loss.item(), loss_CE.item(), loss_MD.item(), loss_NLL.item()))

                    # statistics
                    _, preds = torch.max(logits, 1)

                    epoch_loss = (loss.item() * inputs.size(0) + cnt * epoch_loss) / (cnt + inputs.size(0))
                    epoch_acc = (torch.sum(preds == labels.data) + epoch_acc * cnt).double() / (cnt + inputs.size(0))

                    cnt += inputs.size(0)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                if phase == 'train':
                    scheduler.step()

            print('this epoch takes {} seconds.'.format(time.time() - since2))

    @torch.no_grad()
    def compute_class_prototypes(self, loader):
        self.eval()
        feats = []
        labels = []

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            z = self.embedding(x)
            feats.append(z)
            labels.append(y)

        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)

        prototypes = []
        for k in range(self.num_classes):
            prototypes.append(feats[labels == k].mean(dim=0))
        dist = torch.cdist(feats, feats).pow(2)

        sigma = torch.median(dist).detach()

        self.class_prototypes = torch.stack(prototypes)  # [K, d]
        self.sigma = sigma

    @torch.no_grad()
    def get_ND_scores(self, loader):
        """
        HSIC-based Novelty Detection
        return: ND_scores [N]
        """
        self.eval()

        assert hasattr(self, "class_prototypes"), \
            "Please compute class prototypes before ND detection."

        ND_scores = []

        with torch.no_grad():
            for _, (inputs, _) in enumerate(tqdm(loader)):
                inputs = inputs.to(self.device)

                # -------- feature extraction --------
                z = self.embedding(inputs)  # [B, d]

                # -------- nearest class prototype --------
                sim = F.cosine_similarity(
                    z.unsqueeze(1),  # [B, 1, d]
                    self.class_prototypes.unsqueeze(0),  # [1, K, d]
                    dim=-1
                )  # [B, K]



        ND_scores = np.concatenate(ND_scores, axis=0)

        return ND_scores
