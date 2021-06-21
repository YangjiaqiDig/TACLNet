from argparse import ArgumentParser
from PIL import Image
from lstm import *
from save_history import *
from util import *
import numpy as np
import torch.nn as nn
import time
from topo import *


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        pred = input[:, 1]
        if not (target.size() == pred.size()):
            raise ValueError("Target size ({}) must be the same as pred size ({})"
                             .format(target.size(), pred.size()))

        max_val = (-pred).clamp(min=0)
        loss = pred - pred * target + max_val + \
               ((-max_val).exp() + (-pred - max_val).exp()).log()

        invprobs = F.logsigmoid(-pred * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


def dice_score(input, target, args):
    mid = int(args.step_size / 2)
    predict, label = input[:, mid * 2: (mid + 1) * 2], target[:, mid].to(args.device)

    totalScore = 0
    for i, eachInput in enumerate(predict):
        pred = torch.argmax(eachInput, dim=0).float()
        iflat = pred.contiguous().view(-1)
        tflat = label[i].contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        score = (2. * intersection + 0.000001) / (iflat.sum() + tflat.sum() + 0.000001)
        totalScore += score
    return totalScore / predict.shape[0]


def dice_loss(input, target):
    smooth = 1.

    iflat = input[:, 1].contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    # return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


class MixDiceFocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss_focal = self.focal(input, target)
        loss_dice = dice_loss(input, target)  # torch.log(dice_loss(input, target))

        loss = self.alpha * loss_focal + loss_dice
        # loss = self.alpha*loss_focal - loss_dice
        return loss.mean()


class MixDiceCrossEntropyLoss(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.crossEntropy = nn.CrossEntropyLoss()

    def forward(self, input, target):
        loss_dice = dice_loss(input, target)  # torch.log(dice_loss(input, target))
        loss = self.alpha * self.crossEntropy(input, target) + loss_dice * self.beta
        return loss


def lossForSeqSlices(predicts, labels, args):
    loss_fun_mix = MixDiceCrossEntropyLoss(1., 0.0)
    total_loss = 0
    for i in range(args.step_size):
        total_loss += loss_fun_mix(predicts[:, i * 2: (i + 1) * 2], labels[:, i].to(args.device))

    loss = total_loss / args.step_size
    return loss


def accForSeqSlicesBatch(predicts, labels, args):
    total_acc = 0
    for i in range(args.step_size):
        pred_class = torch.argmax(predicts[:, i * 2: (i + 1) * 2], dim=1)
        total_acc += accuracy_for_batch(labels[:, i].cpu(), pred_class.cpu(), args)

    acc = total_acc / args.step_size

    return acc


def accForSeqSlices(org, predicts, labels, args, batch, i, name):
    total_acc = 0
    likelihoodMaps, pred_class_all = [], []
    for step in range(args.step_size):
        pred_class = torch.argmax(predicts[:, step * 2 : (step + 1) * 2], dim=1)
        pred_class_all.append(pred_class)
        acc = accuracy_check(labels[:, step].cpu(), pred_class.cpu())
        total_acc += acc
        likelihoodMap = predicts[:, step * 2 : (step + 1) * 2][:, 1, :, :]
        likelihoodMaps.append(likelihoodMap)
    
    acc = total_acc / args.step_size	
    save_prediction(likelihoodMaps, pred_class_all, args, batch, i + 1, name)
    save_groundTrue(org.squeeze(2), labels, args, batch, i + 1, name)
    return acc, acc, acc, acc


def train_LSTM_TopoAttention(train_loader, val_loader, args):
    start = time.time()
    logging.info("Start Training CLSTM")
    in_channels = 1
    model = ConvLSTM(input_dim=in_channels, hidden_dim=[16, 8, 2 * args.step_size], kernel_size=(3, 3), num_layers=3,
                     batch_first=True, bias=True, return_all_layers=False).to(args.device)
    print(2 * args.step_size)
    if args.device == "cuda":
        print("GPU: ", torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=list(
            range(torch.cuda.device_count()))).cuda()
    if args.topo_attention:
        path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_models' + args.check_point
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("Start Training Topo Attention, loading pre-trained model from {}".format(path))

    global loss_fun
    # loss_fun = nn.CrossEntropyLoss()
    loss_fun = MixDiceCrossEntropyLoss(1.0, 0.0)
    LR = args.lr_topo if args.topo_attention else args.lr
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)  # , eps=1e-05, weight_decay=0.000001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    iter_attention_train = []
    iter_attention = []
    for i in range(0, args.n_epochs):
        model.train()
        new_epoch_start = time.time()
        for batch, data in enumerate(train_loader):
            images, labels_down = data[0], data[1]  # data[0]: (batch, 3, 1, size, size) label: (batch, 3,  size, size)

            output_down = model(images.to(args.device), None)  # out: [batch, 6, size, size]
            if args.topo_attention:
                output_topo, label, iter_attention_train = topo_attention(output_down, labels_down,
                                                                          iter_attention_train, args, batch, i)
                loss = loss_fun(output_topo, label)  # labels[:,1].to(args.device))
            else:
                loss = lossForSeqSlices(output_down, labels_down, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if not args.topo_attention: lr_scheduler.step()
        model.eval()
        total_acc = 0
        total_loss = 0
        total_score = 0
        for batch, data in enumerate(train_loader):
            images, labels_down = data[0], data[1]  # data[0]: (batch, 3, 1, size, size) label: (batch, 3,  size, size)
            with torch.no_grad():
                output_down = model(images.to(args.device),
                                    None)  # out: [batch, 3, 6, size, size] -> [batch, 6, size, size]
                if args.topo_attention:
                    output_topo, label, iter_attention = topo_attention(output_down, labels_down, iter_attention, args,
                                                                        batch, i)
                    loss = loss_fun(output_topo, label)  # labels[:, 1].to(args.device))
                    pred = torch.argmax(output_topo, dim=1)
                    acc = accuracy_for_batch(label.cpu(), pred.cpu(), args)
                else:
                    loss = lossForSeqSlices(output_down, labels_down, args)
                    acc = accForSeqSlicesBatch(output_down, labels_down, args)
                score = dice_score(output_down, labels_down, args)
                total_acc += acc
                total_loss += loss.cpu().item()
                total_score += score.cpu()
        train_acc_epoch, train_loss_epoch = total_acc / (batch + 1), total_loss / (batch + 1)
        print('Epoch', str(i + 1), 'Train loss:', train_loss_epoch, "Train acc", train_acc_epoch, "Train Dice Score: ",
              total_score / (batch + 1))
        """Validation for every 5 epochs"""
        if (i + 1) % 5 == 0 or args.topo_attention:
            total_val_loss = 0
            total_val_acc = 0
            total_val_score = 0
            total_acc_val_1, total_acc_val_2, total_acc_val_3 = 0, 0, 0
            for batch, data in enumerate(val_loader):
                if args.database == 'Hepatic':
                    images, labels_down, name = data[0], data[1], data[
                        2]  # data[0]: (batch, 3, 1, size, size) label: (batch, 3,  size, size)
                else:
                    images, labels_down = data[0], data[1]
                    name = []
                with torch.no_grad():
                    output_down = model(images.to(args.device),
                                        None)  # out: [batch, 3, 6, size, size] -> [batch, 6, size, size]
                    if args.topo_attention:
                        output_topo, label, iter_attention = topo_attention(output_down, labels_down, iter_attention,
                                                                            args, batch, i + 1, True)
                        loss = loss_fun(output_topo, label)  # labels[:, 1].to(args.device))
                        likelihoodMap = output_topo[:, 1, :, :]
                        pred = torch.argmax(output_topo, dim=1)
                        acc_val = accuracy_check(label.cpu(), pred.cpu())
                        save_prediction_att(likelihoodMap, pred, args, batch, i + 1)
                        save_groundTrue_att(images.squeeze(2), labels_down, args, batch, i + 1)
                    else:
                        loss = lossForSeqSlices(output_down, labels_down.long(), args)
                        acc_val, acc_val_1, acc_val_2, acc_val_3 = accForSeqSlices(images, output_down,
                                                                                   labels_down.long(), args, batch, i,
                                                                                   name)
                    score = dice_score(output_down, labels_down.long(), args)
                    total_val_loss += loss.cpu().item()
                    total_val_acc += acc_val
                    total_val_score += score.cpu()
                    # if not args.topo_attention:
                    #     total_acc_val_1 += acc_val_1
                    #     total_acc_val_2 += acc_val_2
                    #     total_acc_val_3 += acc_val_3
            valid_acc_epoch, valid_loss_epoch = total_val_acc / (batch + 1), total_val_loss / (batch + 1),
            topo_time = time.time() - new_epoch_start if args.topo_attention else 0
            print('Val loss:', valid_loss_epoch, "val acc:", valid_acc_epoch, "val Dice score:",
                  total_val_score / (batch + 1), "topo-attention time", topo_time)
            # if not args.topo_attention:
            #     valid_acc1_epoch, valid_acc2_epoch, valid_acc3_epoch = total_acc_val_1 / (batch + 1), total_acc_val_2 / (batch + 1), total_acc_val_3 / (batch + 1),
            #     print("val acc 1:", valid_acc1_epoch, "val acc 2:", valid_acc2_epoch, "val acc 3:", valid_acc3_epoch)

            header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
            save_values = [i + 1, train_loss_epoch, train_acc_epoch, valid_loss_epoch, valid_acc_epoch]
            export_history(header, save_values, args)

        if (i + 1) % 10 == 0 or args.topo_attention:
            save_models(i + 1, model, optimizer, args)

    print('final running time:', time.time() - start)


if __name__ == "__main__":
    def upsampling(likelihoodMap, times=2):
        up = [nn.Upsample(scale_factor=2, mode="nearest"), nn.Upsample(size=(625, 625), mode="nearest")]

        for i in range(times):
            likelihoodMap = up[i](likelihoodMap)
        return likelihoodMap


    imgPathlh = Image.open('imgCheck/lh_in.png')
    imgPathscore = Image.open('imgCheck/score.png')
    for i, img_as_img in enumerate(ImageSequence.Iterator(imgPathlh)):
        img_as_np = np.asarray(img_as_img)
    img_as_tensor = torch.from_numpy(img_as_np).float()
    img_as_tensor = (img_as_tensor - torch.min(img_as_tensor)) / (torch.max(img_as_tensor) - torch.min(img_as_tensor))
    for i, img_as_img in enumerate(ImageSequence.Iterator(imgPathscore)):
        img_as_np2 = np.asarray(img_as_img)
    img_as_tensor2 = torch.from_numpy(img_as_np2).float()
    img_as_tensor2 = (img_as_tensor2 - torch.min(img_as_tensor2)) / (
            torch.max(img_as_tensor2) - torch.min(img_as_tensor2))
    print(img_as_tensor2.shape)
    img_as_tensor2 = upsampling(img_as_tensor2.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

    result = 0.5 * img_as_tensor + img_as_tensor2
    result = (result - torch.min(result)) / (torch.max(result) - torch.min(result))

    # result[result > 1] = 1
    # result[result < 0] = 0
    # result = result <= 0.5
    saveForTest(result, 0, type='result4')
    print('done')
