import csv
import os
from PIL import Image

import numpy as np
import torch

from topo import *


def export_history(header, save_values, args):
    folder = args.save_folder + '/valid_' + str(args.valid_round)
    if not os.path.exists(folder):
        os.makedirs(folder)
    if args.topo_attention:
        file_name = args.save_folder + '/valid_' + str(args.valid_round) + 'topo_history_Valid.csv'
    else:
        file_name = args.save_folder + '/valid_' + str(args.valid_round) + 'history_Valid.csv'
    file_existence = os.path.isfile(file_name)
    if not file_existence:
        file = open(file_name, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(save_values)
    else:
        file = open(file_name, 'a', newline='')
        writer = csv.writer(file)
        writer.writerow(save_values)
    file.close()


def save_models(epoch, model, optimizer, args):
    if args.topo_attention:
        path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_models_topo'
    else:
        path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_models'
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({
        'epoch': epoch,
        'args': args,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path + "/model_epoch_{0}.pwf".format(epoch))


def save_prediction(likelihood_map, pred_class, args, batch, epoch):
    for i in range(len(likelihood_map)):
        img_as_np, pred_as_np = likelihood_map[i].cpu().data.numpy(), pred_class[i].cpu().data.numpy()

        img_as_np, pred_as_np = img_as_np * 255, pred_as_np * 255
        img_as_np, pred_as_np = img_as_np.astype(np.uint8), pred_as_np.astype(np.uint8)
        # print(img_as_np, img_as_np.shape)
        img, pred = Image.fromarray(img_as_np.squeeze(0)), Image.fromarray(pred_as_np.squeeze(0))
        if args.topo_attention:
            path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images_topo' + '/epoch_' + str(
                epoch) + '/'
        else:
            path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images' + '/epoch_' + str(epoch) + '/'

        if not os.path.exists(path):
            os.makedirs(path)
        # SAVE Valid Likelihood Images and Prediction
        export_name_lh = str(batch) + '-{}lh.png'.format(i)
        export_name_pred = str(batch) + '-{}pred.png'.format(i)
        img.save(path + export_name_lh)
        pred.save(path + export_name_pred)


def save_groundTrue(images, labels, args, batch, epoch):
    for i in range(images.shape[1]):
        img_as_np = images[:,i].cpu().data.numpy()
        label_as_np = labels[:,i].cpu().data.numpy()

        img_as_np, label_as_np = img_as_np * 255, label_as_np * 255
        img_as_np, label_as_np = img_as_np.astype(np.uint8), label_as_np.astype(np.uint8)
        # print(img_as_np, img_as_np.shape)

        img, label = Image.fromarray(img_as_np.squeeze(0)), Image.fromarray(label_as_np.squeeze(0))
        if args.topo_attention:
            path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images_topo' + '/epoch_' + str(
                epoch) + '/'
        else:
            path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images' + '/epoch_' + str(epoch) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        # SAVE Valid ground true Images
        export_name_orig = str(batch) + '-{}orig.png'.format(i)
        export_name_gt = str(batch) + '-{}gt.png'.format(i)
        img.save(path + export_name_orig)
        label.save(path + export_name_gt)
