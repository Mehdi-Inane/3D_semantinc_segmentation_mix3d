from dataset.dataset import S3DISDataset
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset.merge import SimpleCollateMergeToTensor
from models.pointtransformer import PointNetSegHead
from config import Config
from models.loss import PointNetSegLoss
import torch.optim as optim
import time 
def compute_iou(targets, predictions):

    targets = targets.reshape(-1)
    predictions = predictions.reshape(-1)

    intersection = torch.sum(predictions == targets) # true positives
    union = len(predictions) + len(targets) - intersection

    return intersection / union 



cfg = {'num_points' : 128,
            'batch_size': 4,
            'use_normals': False,
            'optimizer': 'RangerVA',
            'lr': 0.005,
            'decay_rate': 0.0001,
            'lr_decay': 0.7,
            'epochs': 500,
            'num_classes': 13,
            'num_part': 50,
            'dropout': 0.3,
            'M': 5,
            'K': 8,
            'd_m': 256,
            'dd_m': 64,
            'step_size': 15,
    }
data_path = 'data/data/processed_s3dis_normals'
train_dataset = S3DISDataset(data_path)
NUM_TRAIN_POINTS = 2500 # train/valid points
NUM_TEST_POINTS = 15000
DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')
train_collate = SimpleCollateMergeToTensor(device=DEVICE,downsample_to=int(NUM_TRAIN_POINTS/2))
train_dataloader = DataLoader(train_dataset,batch_size=2,collate_fn=train_collate)
test_dataset = S3DISDataset(data_path,mode='test')
test_dataloader = DataLoader(test_dataset,batch_size = 2)
valid_dataset = S3DISDataset(data_path,mode='val')
valid_dataloader = DataLoader(valid_dataset,batch_size = 2,collate_fn=train_collate)

########################### Model########################"
seg_model = PointNetSegHead(num_points=NUM_TRAIN_POINTS,m=14)
EPOCHS = 20
LR = 0.0001
alpha = np.ones(14)
alpha[0:3] *= 0.25 # balance background classes
alpha[-1] *= 0.75  # balance clutter class

gamma = 1

optimizer = optim.Adam(seg_model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3, 
                                              step_size_up=1000, cycle_momentum=False)
criterion = PointNetSegLoss(alpha=alpha, gamma=gamma, dice=True).to(DEVICE) # Focal loss
seg_model = seg_model.to(DEVICE)


train_loss = []
train_accuracy = []
train_iou = []
valid_loss = []
valid_accuracy = []
valid_iou = []
BATCH_SIZE = 2

# stuff for training
num_train_batch = int(np.ceil(len(train_dataset)/BATCH_SIZE))
num_valid_batch = int(np.ceil(len(valid_dataset)/BATCH_SIZE))

for epoch in range(1, EPOCHS + 1):
    # place model in training mode
    seg_model = seg_model.train()
    _train_loss = []
    _train_accuracy = []
    _train_iou = []
    for i, batch in enumerate(train_dataloader, 0):
        print("back here")
        points,feats,targets = batch[0]
        points = points.unsqueeze(0).transpose(2, 1).to(DEVICE)
        targets = targets.to(DEVICE).unsqueeze(0)
        
        # zero gradients
        optimizer.zero_grad()
        
        # get predicted class logits
        preds, _, _ = seg_model(points)
        # get class predictions
        pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)
        # get loss and perform backprop
        loss = criterion(preds, targets, pred_choice) 
        loss.backward()
        optimizer.step()
        scheduler.step() # update learning rate
        # get metrics
        correct = pred_choice.eq(targets.data).cpu().sum()
        accuracy = correct/float(BATCH_SIZE*NUM_TRAIN_POINTS)
        iou = compute_iou(targets, pred_choice)

        # update epoch loss and accuracy
        _train_loss.append(loss.item())
        _train_accuracy.append(accuracy)
        _train_iou.append(iou.item())

        if i % 100 == 0:
            print(f'\t [{epoch}: {i}/{num_train_batch}] ' \
                  + f'train loss: {loss.item():.4f} ' \
                  + f'accuracy: {accuracy:.4f} ' \
                  + f'iou: {iou:.4f}')
    train_loss.append(np.mean(_train_loss))
    train_accuracy.append(np.mean(_train_accuracy))
    train_iou.append(np.mean(_train_iou))

    print(f'Epoch: {epoch} - Train Loss: {train_loss[-1]:.4f} ' \
          + f'- Train Accuracy: {train_accuracy[-1]:.4f} ' \
          + f'- Train IOU: {train_iou[-1]:.4f}')

    # pause to cool down
    # get test results after each epoch
    with torch.no_grad():

        # place model in evaluation mode
        seg_model = seg_model.eval()

        _valid_loss = []
        _valid_accuracy = []
        _valid_iou = []
        for i, (points, targets) in enumerate(valid_dataloader, 0):

            points = points.transpose(2, 1).to(DEVICE)
            targets = targets.squeeze().to(DEVICE)

            preds, _, A = seg_model(points)
            pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)

            loss = criterion(preds, targets, pred_choice) 

            # get metrics
            correct = pred_choice.eq(targets.data).cpu().sum()
            accuracy = correct/float(BATCH_SIZE*NUM_TRAIN_POINTS)
            iou = compute_iou(targets, pred_choice)

            # update epoch loss and accuracy
            _valid_loss.append(loss.item())
            _valid_accuracy.append(accuracy)
            _valid_iou.append(iou.item())

            if i % 100 == 0:
                print(f'\t [{epoch}: {i}/{num_valid_batch}] ' \
                  + f'valid loss: {loss.item():.4f} ' \
                  + f'accuracy: {accuracy:.4f} '
                  + f'iou: {iou:.4f}')
        
        valid_loss.append(np.mean(_valid_loss))
        valid_accuracy.append(np.mean(_valid_accuracy))
        valid_iou.append(np.mean(_valid_iou))
        print(f'Epoch: {epoch} - Valid Loss: {valid_loss[-1]:.4f} ' \
              + f'- Valid Accuracy: {valid_accuracy[-1]:.4f} ' \
              + f'- Valid IOU: {valid_iou[-1]:.4f}')


    # save best models
    if valid_iou[-1] >= best_iou:
        best_iou = valid_iou[-1]
        torch.save(seg_model.state_dict(), f'checkpoints/seg_model_{epoch}.pth')

    # if valid_mcc[-1] >= best_mcc:
    #     best_mcc = valid_mcc[-1]