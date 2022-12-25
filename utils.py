import os
import random
import pickle
from math import e
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

import albumentations as A
from albumentations.augmentations import functional as albumentations_F
from albumentations.pytorch import ToTensorV2

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
from matplotlib import cm
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from model import RetrievalModel, CosFaceLoss, ArcFaceLoss, GradualWarmupScheduler
from data import split_data, RetrievalDataset, preprocess_data


def set_seed(seed):
    '''Set a random seed for complete reproducibility.'''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def train(dataloader, model, loss_function, optimizer, scheduler, scaler, mixed_precision=False, device='cuda'):
    ''' Training loop '''

    total_loss = 0.0

    model.train()

    for step, batch in enumerate(tqdm(dataloader)):
        inputs, labels = batch['data'].to(device), batch['labels'].to(device)

        optimizer.zero_grad()

        if mixed_precision:
            with torch.cuda.amp.autocast():
                logits, outputs, _ = model(inputs.float())

                loss = loss_function(logits, labels)
        else:
            if loss_function.__class__.__name__ == 'CrossEntropyLoss':
                outputs, logits, _ = model(inputs.float())
            else:
                logits, outputs, _ = model(inputs.float())

            loss = loss_function(outputs, labels)
        
        total_loss += loss.item()

        if mixed_precision:
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    
    scheduler.step()

    print('Learning rate:', optimizer.param_groups[0]['lr'])

    return total_loss / len(dataloader)


def validate(dataloader, model, loss_function, epoch, device='cuda', plot_embeddings=False):
    ''' Validation loop '''

    total_loss = 0.0
    total_score = 0.0

    model.eval()

    val_embeddings = torch.empty((0))
    val_labels = torch.empty((0))
    val_ids = torch.empty((0))
    
    proj_embeddings = torch.empty((0))

    for step, batch in enumerate(tqdm(dataloader)):
        inputs, labels, ids = batch['data'].to(device), batch['labels'].to(device), batch['ids']

        with torch.no_grad():
            if loss_function.__class__.__name__ == 'CrossEntropyLoss':
                outputs, logits, proj_outputs = model(inputs.float())
            else:
                logits, outputs, proj_outputs = model(inputs.float())

        # Accumulate embeddings
        if loss_function.__class__.__name__ == 'CrossEntropyLoss':
            val_embeddings = torch.cat((val_embeddings, F.normalize(logits, p=2, dim=1).cpu().detach()))
        else:
            val_embeddings = torch.cat((val_embeddings, F.normalize(outputs, p=2, dim=1).cpu().detach()))
        val_labels = torch.cat((val_labels, labels.cpu().detach()))
        val_ids = torch.cat((val_ids, ids))
        
        if plot_embeddings:
            proj_embeddings = torch.cat((proj_embeddings, F.normalize(proj_outputs, p=2, dim=1).cpu().detach()))

        loss = loss_function(outputs, labels)
        total_loss += loss.item()

    val_embeddings, val_labels, val_ids = val_embeddings.numpy(), val_labels.numpy(), val_ids.numpy()

    # Visualize embeddings
    if plot_embeddings:
        visualize_embeddings(proj_embeddings.numpy(), val_labels, epoch)

    return total_loss / len(dataloader), val_embeddings, val_labels, val_ids


def visualize_embeddings(embeddings, labels, epoch):
    ''' Plot normalized embedding on a sphere '''

    num_categories = 12

    # Clip number of samples in embeddings
    embeddings = embeddings[:4000, :]
    labels = labels[:4000]

    cmap = cm.get_cmap('tab20')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for label in range(num_categories):
        indices = labels == label
        ax.scatter(embeddings[indices, 0], embeddings[indices, 1], embeddings[indices, 2], c=np.array(cmap(label)).reshape(1, 4), label = label, alpha=0.5)

    plt.savefig(f'visualizations/embeddings_epoch_{epoch}.png')


class EmbeddingsSearch():
    def __init__(self, model, embeddings, ids, labels, num_classes=12):
        '''
            model: PyTorch model to extract embeddings with
            embeddings - [N, E]: embeddings extracted from OOF data
            ids - [N]: list of image ids from OOF data
            labels - [N]: list of labels from OOF data
            search_data - pd.DataFrame: data to acquire images from (typically a validation set)
            num_classes - int: number of groups to cluster data by
        '''

        super(EmbeddingsSearch, self).__init__()

        self.model = model

        self.nn = NearestNeighbors(n_neighbors=num_classes)
        self.nn.fit(embeddings)

        self.ids = ids
        self.labels = labels

    def embeddings_image_search(self, img, n_neighbors=5):
        ''' Retrieve closest samples to a selected image  '''

        # Preprocess image
        img = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
        img = albumentations_F.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 255.0)
        img = torch.tensor(img, dtype=torch.float32, device='cuda')
        img = img.permute(2, 0, 1).unsqueeze(0)

        # Extract embedding from the image
        self.model.eval()
        with torch.no_grad():
            _, outputs, _ = self.model(img) 
            outputs = F.normalize(outputs, p=2, dim=1)
            outputs = outputs.cpu().detach().numpy()

        # Find closest matches
        neighbors = self.nn.kneighbors(outputs, n_neighbors + 1, return_distance=False)[0][1:]

        retrieved_ids = []; retrieved_labels = []
        for x in neighbors:
            retrieved_ids.append(int(self.ids[x]))
            retrieved_labels.append(int(self.labels[x]))

        return retrieved_ids, retrieved_labels

    def evaluate_search(self, search_data):
        ''' Evaluate search on a selected dataset '''

        tp_3 = 0; fp_3 = 0;
        tp_5 = 0; fp_5 = 0;
        tp_10 = 0; fp_10 = 0;

        for i, row in search_data.iterrows():
            img_path = 'data/stanford_image_retrieval/' + row['path']
            img = cv2.imread(img_path)

            label = row['class_id']

            retrieved_ids, retrieved_labels = self.embeddings_image_search(img, n_neighbors=10)
            if label in retrieved_labels:
                tp_10 += 1
            else:
                fp_10 += 1

            if label in retrieved_labels[:5]:
                tp_5 += 1
            else:
                fp_5 += 1

            if label in retrieved_labels[:3]:
                tp_3 += 1
            else:
                fp_3 += 1

        recall_at_k = {
            'k3': tp_3 / (tp_3 + fp_3),
            'k5': tp_5 / (tp_5 + fp_5),
            'k10': tp_10 / (tp_10 + fp_10)
        }

        return recall_at_k


def run():
    ''' Main function '''

    ### PARAMS ###
    seed = 100
    num_classes = 22634 
    training_folds = [0]
    num_epochs = 20
    device = 'cuda'
    mixed_precision = False
    plot_embeddings = False

    if plot_embeddings:
        # Show embeddings on super_class_id
        num_classes = 12 
    ### PARAMS ###

    set_seed(seed)

    data = pd.read_csv('data/stanford_image_retrieval/Ebay_info.txt', sep=' ')
    data = preprocess_data(data)
    print(data)

    train_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])
    val_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

    backbone = timm.create_model('resnet18', pretrained=True, num_classes=0, in_chans=3)
    model = RetrievalModel(backbone, num_classes, is_emb_proj=plot_embeddings).to(device)

    loss_function = CosFaceLoss(in_features=backbone.num_features, out_features=num_classes, s=30.0, m=0.3)
    #loss_function = ArcFaceLoss(in_features=backbone.num_features, out_features=num_classes, s=20.0, m=0.5)
    #loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs // 4)
    scaler = torch.cuda.amp.GradScaler()

    for current_fold in training_folds:
        train_data, val_data = split_data(data, current_fold=current_fold)

        train_dataset = RetrievalDataset(train_data, train_transforms, plot_embeddings)
        val_dataset = RetrievalDataset(val_data, val_transforms, plot_embeddings)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        best_loss = np.inf
        best_metric = 0
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}')

            train_loss = train(train_loader, model, loss_function, optimizer, scheduler, scaler, mixed_precision=mixed_precision, device=device)
            val_loss, val_embeddings, val_labels, val_ids = validate(val_loader, model, loss_function, epoch + 1, device='cuda', plot_embeddings=plot_embeddings)

            embeddings_search = EmbeddingsSearch(model, val_embeddings, val_ids, val_labels, num_classes=num_classes)
            recall_at_k = embeddings_search.evaluate_search(val_data[:3000])

            print(f'Train loss: {train_loss}')
            print(f'Val loss: {val_loss}, Val score K=3: {recall_at_k["k3"]} K=5: {recall_at_k["k5"]} K=10: {recall_at_k["k10"]}')

            # Save a checkpoint
            if best_metric < recall_at_k['k5']:
                print('New Record')

                best_metric = recall_at_k['k5']
                torch.save(model.state_dict(), f'checkpoints/weights_epoch_{epoch + 1}.pt') 

                # Save OOF embeddings and ids
                val_embeddings = {
                    'embeddings': val_embeddings,
                    'labels': val_labels,
                    'ids': val_ids
                }

                with open(f'embeddings/embeddings_oof_fold_{current_fold + 1}.pickle', 'wb') as f:
                    pickle.dump(val_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
