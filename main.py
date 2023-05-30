import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F
from model import Model
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataListLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Batch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils import *
import numpy as np
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=2022, help='random seed')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--dataset', type=str, default='AIDS', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--device', type=str, default='cuda:3', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--agument_num', type=int, default=7, help='the number of agumentation')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)

args.num_features = dataset.num_features

args.flip = True

print(args)

kfold = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)

def k_fold_cross_val_train():
    y = dataset.data.y.cpu().detach().numpy()

    num_ones = np.sum(y==1)
    num_zeros = np.sum(y==0)

    # We choose the minority class as anomaly. Abnormal: 1; Normal: 0

    if num_ones < num_zeros:
        args.flip = False
    else:
        args.flip = True

    results = {}

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset, dataset.data.y)):
        print('Fold ' + str(fold))
        print('-' * 30)

        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        trainloader = DataListLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        testloader = DataListLoader(dataset, batch_size=args.batch_size, sampler=test_subsampler)

        model = Model(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_auc = train(model, optimizer, trainloader, testloader)
        
        results[fold] = best_auc
        
        print('Test set results, auc score = {:.6f}'.format(best_auc))

    return results


def train(model, optimizer, train_loader, test_loader):
    min_loss = 1e10
    patience_cnt = 0
    loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    best_auc = 0
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = random_agumentation(data)
            for element in x:
                if hasattr(element, 'edge_attr'):
                    element.edge_attr = None
                else:
                    element.edge_attr = None
            x = Batch.from_data_list(x).to(args.device)
            y = torch.from_numpy(np.array(y, dtype=np.int64)).to(args.device)
            out = model(x)
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(y).sum().item()
        acc_train = correct / len(train_loader.dataset)

        auc_score = test(model, test_loader)
        if auc_score > best_auc:
            best_auc = auc_score

        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
              'acc_train: {:.6f}'.format(acc_train), 'AUC: {:.6f}'.format(best_auc), 'time: {:.6f}s'.format(time.time() - t))

        loss_values.append(loss_train)
        torch.save(model.state_dict(), '{}.pth'.format(epoch))
        
        if loss_values[-1] < min_loss:
            min_loss = loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_auc


def test(model, loader):
    model.eval()
    loss = 0.0
    cnt = 0
    gt = None
    pred = None
    for data in loader:
        cnt += 1
        score = 0.0
        for idx in range(7):
            data_sag, y = specific_agumentation(data, idx)
            for element in data_sag:
                if hasattr(element, 'edge_attr'):
                    element.edge_attr = None
                else:
                    element.edge_attr = None
            data_aug = Batch.from_data_list(data_sag)
            data_aug = data_aug.to(args.device)
            if args.flip:
                label = 1 - data_aug.y.cpu().detach().numpy()
            else:
                label = data_aug.y.cpu().detach().numpy()
            out = model.inference(data_aug)
            temp = negative_entropy_score(out.cpu().detach().numpy())
            score += temp
        score = score / 7

        if cnt == 1:
            pred = score
            gt = label
        else:
            pred = np.concatenate((pred, score))
            gt = np.concatenate((gt, label))

    auc_score = roc_auc_score(gt, pred)

    if auc_score < 0.5:
        auc_score = 1 - auc_score

    return auc_score


if __name__ == '__main__':
    results = k_fold_cross_val_train()

    print('\nK-fold cross validation results:')
    print('-' * 30)
    auc_sum = 0.0
    for key, value in results.items():
        print('Fold ' + str(key) + ': ' + str(value))
        auc_sum += value
    auc = np.array(list(results.values()))
    print('\nAverage results : ' + str(auc.mean()) + ', std: ' + str(auc.std()))

