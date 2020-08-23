from argparse import ArgumentParser
from PIL import Image
from lstm import *
from model import UNET
from save_history import *
from util import *
import numpy as np
import torch.nn as nn
import time
from topo import  *
import convolutional_rnn

softmax = nn.Softmax2d()

def train_LSTM_TopoAttention(train_loader, val_loader, args):
    start = time.time()
    logging.info("Start Training CLSTM")
    in_channels = 1
    model = ConvLSTM(input_dim=in_channels, hidden_dim=[64, 32, 8, 6], kernel_size=(5, 5), num_layers=4,
                     batch_first=True, bias=True, return_all_layers=False).to(args.device)
    if args.device == "cuda":
        print("GPU: ", torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=list(
            range(torch.cuda.device_count()))).cuda()
    if args.topo_attention:
        path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_models' + args.check_point
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("Start Training Topo Attention, loading pre-trained model from {}".format(path))


    loss_fun = nn.CrossEntropyLoss()
    # loss_fun_att = nn.MSELoss()
    LR = args.lr_topo if args.topo_attention else args.lr
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    for i in range(0, args.n_epochs):
        """Train each epoch"""
        model.train()
        # iter_attention = torch.tensor(0)
        for batch, data in enumerate(train_loader):
            images, labels = data[0], data[1]  #image: (batch, 3, 1, size, size) label: (batch, 3,  size, size)
            out, h = model(images.to(args.device), None) # out: [(batch, 3, 6, size, size)] -> [get last hidden out]
            output = out[0][:,-1,:,:,:]  # torch.Size([batch, 6, 1024, 1024]) -> only keep last step hidden
            if args.topo_attention:
                output = topo_attention(output, labels, args)
                loss = loss_fun(output, labels[:,1].to(args.device))
            else:
                loss_1 = loss_fun(output[:,:2], labels[:,0].to(args.device))
                loss_2 = loss_fun(output[:,2:4], labels[:,1].to(args.device))
                loss_3 = loss_fun(output[:,-2:], labels[:,2].to(args.device))
                loss = (loss_1 + loss_2 + loss_3) / 3
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # lr_scheduler.step()
        model.eval()
        total_acc = 0
        total_loss = 0
        for batch, data in enumerate(train_loader):
            images, labels = data[0], data[1]
            with torch.no_grad():
                out, h = model(images.to(args.device), None)
                output = out[0][:, -1, :, :, :]  # torch.Size([batch, 6, 1024, 1024]) -> only keep last step hidden
                if args.topo_attention:
                    output = topo_attention(output, labels, args)
                    loss = loss_fun(output, labels[:, 1].to(args.device))
                    pred = torch.argmax(output, dim=1)
                    acc = accuracy_for_batch(labels[:, 1].cpu(), pred.cpu(), args)
                else:
                    loss_1 = loss_fun(output[:, :2], labels[:, 0].to(args.device))
                    pred_class_1 = torch.argmax(output[:, :2], dim=1)
                    acc_1 = accuracy_for_batch(labels[:, 0].cpu(), pred_class_1.cpu(), args)

                    loss_2 = loss_fun(output[:, 2:4], labels[:, 1].to(args.device))
                    pred_class_2 = torch.argmax(output[:, 2:4], dim=1)
                    acc_2 = accuracy_for_batch(labels[:, 1].cpu(), pred_class_2.cpu(), args)

                    loss_3 = loss_fun(output[:, -2:], labels[:, 2].to(args.device))
                    pred_class_3 = torch.argmax(output[:, -2:], dim=1)
                    acc_3 = accuracy_for_batch(labels[:, 2].cpu(), pred_class_3.cpu(), args)

                    loss = (loss_1 + loss_2 + loss_3) / 3
                    acc = (acc_1 + acc_2 + acc_3) / 3

                total_acc += acc
                total_loss += loss.cpu().item()
        train_acc_epoch, train_loss_epoch = total_acc / (batch + 1), total_loss / (batch + 1)
        print('Epoch', str(i + 1), 'Train loss:', train_loss_epoch, "Train acc", train_acc_epoch)
        """Validation for every 5 epochs"""
        if (i + 1) % 5 == 0 or args.topo_attention:
            total_val_loss = 0
            total_val_acc = 0
            for batch, data in enumerate(val_loader):
                images, labels = data[0], data[1]
                with torch.no_grad():
                    out, h = model(images.to(args.device))
                    output = out[0][:, -1, :, :, :]
                    if args.topo_attention:
                        output = topo_attention(output, labels, args, batch, i + 1, True)
                        loss = loss_fun(output, labels[:, 1].to(args.device))
                        likelihoodMap = softmax(output)[:, 1, :, :]
                        pred = torch.argmax(output, dim=1)
                        acc_val = accuracy_check(labels[:, 1].cpu(), pred.cpu())
                        save_prediction_att(likelihoodMap, pred, args, batch, i + 1)
                        save_groundTrue_att(data[0].squeeze(2), labels, args, batch, i + 1)
                    else:
                        loss_1 = loss_fun(output[:, :2], labels[:, 0].to(args.device))
                        likelihoodMap_1 = softmax(output[:, :2])[:, 1, :, :]
                        pred_class_1 = torch.argmax(output[:, :2], dim=1)
                        acc_val_1 = accuracy_check(labels[:, 0].cpu(), pred_class_1.cpu())

                        loss_2 = loss_fun(output[:, 2:4], labels[:, 1].to(args.device))
                        likelihoodMap_2 = softmax(output[:, 2:4])[:, 1, :, :]
                        pred_class_2 = torch.argmax(output[:, 2:4], dim=1)
                        acc_val_2 = accuracy_check(labels[:, 1].cpu(), pred_class_2.cpu())

                        loss_3 = loss_fun(output[:, -2:], labels[:, 2].to(args.device))
                        likelihoodMap_3 = softmax(output[:, -2:])[:, 1, :, :]
                        pred_class_3 = torch.argmax(output[:, -2:], dim=1)
                        acc_val_3 = accuracy_check(labels[:, 2].cpu(), pred_class_3.cpu())

                        loss = (loss_1 + loss_2 + loss_3) / 3
                        acc_val = (acc_val_1 + acc_val_2 + acc_val_3) / 3

                        save_prediction([likelihoodMap_1, likelihoodMap_2, likelihoodMap_3], [pred_class_1, pred_class_2, pred_class_3], args, batch, i + 1)
                        save_groundTrue(data[0].squeeze(2), labels, args, batch, i + 1)
                    total_val_loss += loss.cpu().item()
                    total_val_acc += acc_val
            valid_acc_epoch, valid_loss_epoch = total_val_acc / (batch + 1), total_val_loss / (batch + 1),
            print('Val loss:', valid_loss_epoch, "val acc:", valid_acc_epoch)

            header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
            save_values = [i + 1, train_acc_epoch, train_loss_epoch, valid_acc_epoch, valid_loss_epoch]
            export_history(header, save_values, args)

        if (i + 1) % 10 == 0 or args.topo_attention:
            save_models(i + 1, model, optimizer, args)

    print(time.time()-start)

def train_LSTM(args, train_loader, val_loader):
    logging.info("Start Training CLSTM with TOPO input")
    in_channels = 1
    net = ConvLSTM(input_dim=in_channels, hidden_dim=[16, 2, 2], kernel_size=(3,3), num_layers=3,
                 batch_first=True, bias=True, return_all_layers=False).to(args.device)
    if args.device == "cuda":
        print("GPU: ", torch.cuda.device_count())
        net = torch.nn.DataParallel(net, device_ids=list(
            range(torch.cuda.device_count()))).cuda()
    # loss_fun = nn.MSELoss()
    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    for i in range(0, args.n_epochs):
        """Train each epoch"""
        net.train()
        for batch, data in enumerate(train_loader):
            images, labels = data[0], data[1] #image: (batch, 3, 2, size, size) label: (4, size, size)
            inputToModelOnlyLh = images[:,:,1:,:,:]
            out, h = net(inputToModelOnlyLh.to(args.device), None)
            # print(len(out), len(h), out[0].shape, len(h[0]), h[0][0].shape, h[0][1].shape)
            # print(out.shape,len(h), h[0].shape) # torch.Size([4, 3, 2, 1024, 1024]) 2 torch.Size([8, 2, 1, 1024, 1024])
            output = out[0][:,-1,:,:,:] # torch.Size([4, 2, 1024, 1024])
            loss = loss_fun(output, labels.to(args.device))
            net.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # lr_scheduler.step()
        """Get Loss and Accuracy for each epoch"""
        net.eval()
        total_acc = 0
        total_loss = 0
        for batch, data in enumerate(train_loader):
            images, labels = data[0], data[1]
            inputToModelOnlyLh = images[:,:,1:,:,:]
            with torch.no_grad():
                out, h = net(inputToModelOnlyLh.to(args.device), None)
                output = out[0][:, -1, :, :, :]

                loss = loss_fun(output, labels.to(args.device))
                pred_class = torch.argmax(output,dim=1)
                acc = accuracy_for_batch(labels.cpu(), pred_class.cpu(), args)
                total_acc += acc
                total_loss += loss.cpu().item()
        train_acc_epoch, train_loss_epoch = total_acc / (batch + 1), total_loss / (batch + 1)
        print('Epoch', str(i + 1), 'Train loss:', train_loss_epoch, "Train acc", train_acc_epoch)
        """Validation for every 5 epochs"""
        if (i + 1) % 5 == 0:
            total_val_loss = 0
            total_val_acc = 0
            for batch, data in enumerate(val_loader):
                images, labels = data[0], data[1]
                inputToModelOnlyLh = images[:, :, 1:, :, :]
                with torch.no_grad():
                    out, h = net(inputToModelOnlyLh.to(args.device))
                    output = out[0][:, -1, :, :, :]
                    loss = loss_fun(output, labels.to(args.device))
                    pred_class = torch.argmax(output, dim=1)
                    likelihoodMap = softmax(output)[:, 1, :, :]
                    save_prediction(likelihoodMap, pred_class, args, batch, i + 1)
                    save_groundTrue(inputToModelOnlyLh[:,1].squeeze(1).squeeze(1), labels, args, batch, i + 1)
                    total_val_loss += loss.cpu().item()
                    acc_val = accuracy_check(labels.cpu(), pred_class.cpu())
                    total_val_acc += acc_val
            valid_acc_epoch, valid_loss_epoch = total_val_acc / (batch + 1), total_val_loss / (batch + 1),
            print('Val loss:', valid_loss_epoch, "val acc:", valid_acc_epoch)

            header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
            save_values = [i + 1, train_acc_epoch, train_loss_epoch, valid_acc_epoch, valid_loss_epoch]
            export_history(header, save_values, args)

        if (i + 1) % 10 == 0:
            save_models(i + 1, net, optimizer, args)
    return


def train_UNET(args, train_loader, val_loader):
    logging.info("---------Using device %s--------", args.device)

    model = UNET()
    if args.device == "cuda":
        print("GPU: ", torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=list(
            range(torch.cuda.device_count()))).cuda()
    # loss_fun = nn.MSELoss()
    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    logging.info("---------Initializing Training For UNET!--------")
    for i in range(0, args.n_epochs):
        """Train each epoch"""
        model.train()
        for batch, data in enumerate(train_loader):
            images, labels = data[0].unsqueeze(1), data[1]
            output, likelihoodMap = model(images.to(args.device))
            # print(likelihoodMap)
            loss = loss_fun(output, labels.to(args.device))
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        """Get Loss and Accuracy for each epoch"""
        model.eval()
        total_acc = 0
        total_loss = 0
        for batch, data in enumerate(train_loader):
            images, labels = data[0].unsqueeze(1), data[1]
            with torch.no_grad():
                output, likelihoodMap = model(images.to(args.device))
                loss = loss_fun(output, labels.to(args.device))
                pred_class = likelihoodMap >= 0.5
                acc = accuracy_for_batch(labels.cpu(), pred_class.cpu(), args)
                total_acc += acc
                total_loss += loss.cpu().item()
        train_acc_epoch, train_loss_epoch = total_acc / (batch + 1), total_loss / (batch + 1)
        print('Epoch', str(i + 1), 'Train loss:', train_loss_epoch, "Train acc", train_acc_epoch)

        """Validation for every 5 epochs"""
        if (i + 1) % 5 == 0:
            total_val_loss = 0
            total_val_acc = 0
            for batch, data in enumerate(val_loader):
                images, labels = data[0].unsqueeze(1), data[1]
                with torch.no_grad():
                    output, x = model(images.to(args.device))
                    pred_class = likelihoodMap >= 0.5
                    loss = loss_fun(output, labels.to(args.device))
                    save_prediction(likelihoodMap, pred_class, args, batch, i + 1)
                    save_groundTrue(data[0], labels, args, batch, i + 1)
                    total_val_loss += loss.cpu().item()
                    acc_val = accuracy_check(labels.cpu(), pred_class.cpu())
                    total_val_acc += acc_val
            valid_acc_epoch, valid_loss_epoch = total_val_acc / (batch + 1), total_val_loss / (batch + 1),
            print('Val loss:', valid_loss_epoch, "val acc:", valid_acc_epoch)

            header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
            save_values = [i + 1, train_acc_epoch, train_loss_epoch, valid_acc_epoch, valid_loss_epoch]
            export_history(header, save_values, args)

        if (i + 1) % 10 == 0:
            save_models(i + 1, model, optimizer, args)
