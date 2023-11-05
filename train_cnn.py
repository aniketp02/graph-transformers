from models.cnn.cnn import CNN
from config.config import CNNTrainingOptions
from torch.utils.data import DataLoader
from datasets.dataset_patternNet import PatternNetDataset
from datasets.dataset_eurosat import EuroSatDataset


import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F

if __name__ == "__main__":
    # load config
    opt = CNNTrainingOptions().parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)

    # load training data
    if opt.dataset_name == 'PatternNet':
        train_data = PatternNetDataset(opt.filelists , opt.data_root)
        net = CNN(input_channels=3).cuda()
    elif opt.dataset_name == 'EuroSAT':
        train_data = EuroSatDataset(opt.filelists , opt.data_root)
        net = CNN(input_channels=13).cuda()
    else:
        raise("Dataset Not Supported!")
    
    training_data_loader = DataLoader(dataset=train_data,  batch_size=opt.batch_size, shuffle=True)
    train_data_length = len(training_data_loader)
    
    # setup optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # set criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # start train
    for epoch in range(opt.epochs):
        running_loss = 0.0
        net.train()
        for iteration, data in enumerate(training_data_loader):
            # forward
            true_label, img = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(img)
            
            loss = criterion(outputs, true_label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if iteration % 20 == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {iteration + 1:5d}] loss: {running_loss / 2000:.5f}')
                running_loss = 0.0

        # checkpoint
        if epoch %  opt.checkpoint == 0:
            if not os.path.exists(opt.checkpoint_path):
                os.mkdir(opt.checkpoint_path)
            model_out_path = os.path.join(opt.checkpoint_path, 'net_cnn_model_epoch_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(states, model_out_path)
            print("Checkpoint saved to {}".format(epoch))