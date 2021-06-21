import csv
import os
from PIL import Image

import numpy as np
import torch

# from topo import *


def export_history(header, save_values, args):
    folder = args.save_folder + '/valid_' + str(args.valid_round)
    if not os.path.exists(folder):
        os.makedirs(folder)
    if args.topo_attention:
        iter = '_iter' if args.topo_iteration else ''
        file_name = args.save_folder + '/valid_' + str(args.valid_round) + iter + 'topo_history_Valid.csv'
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
        iter = '_iter' if args.topo_iteration else ''
        path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_models_topo' + iter
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


def save_prediction_att(likelihood_map, pred_class, args, batch, epoch):
    img_as_np, pred_as_np = likelihood_map.cpu().data.numpy(), pred_class.cpu().data.numpy()

    img_as_np, pred_as_np = img_as_np * 255, pred_as_np * 255
    img_as_np, pred_as_np = img_as_np.astype(np.uint8), pred_as_np.astype(np.uint8)
    # print(img_as_np, img_as_np.shape)
    img, pred = Image.fromarray(img_as_np.squeeze(0)), Image.fromarray(pred_as_np.squeeze(0))
    if args.topo_attention:
        iter = '_iter' if args.topo_iteration else ''
        path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images_topo'  + iter + '/epoch_' + str(
            epoch) + '/'
    else:
        path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images' + '/epoch_' + str(epoch) + '/'

    if not os.path.exists(path):
        os.makedirs(path)
    # SAVE Valid Likelihood Images and Prediction
    export_name_lh = str(batch) + 'lh.png'
    export_name_pred = str(batch) + 'pred.png'
    img.save(path + export_name_lh)
    pred.save(path + export_name_pred)

def save_prediction(likelihood_map, pred_class, args, batch, epoch, name):
    for i in range(len(likelihood_map)):
        img_as_np, pred_as_np = likelihood_map[i].cpu().data.numpy(), pred_class[i].cpu().data.numpy()

        img_as_np, pred_as_np = img_as_np * 255, pred_as_np * 255
        img_as_np, pred_as_np = img_as_np.astype(np.uint8), pred_as_np.astype(np.uint8)
        # print(img_as_np, img_as_np.shape)
        img, pred = Image.fromarray(img_as_np.squeeze(0)), Image.fromarray(pred_as_np.squeeze(0))
        if args.topo_attention:
            iter = '_iter' if args.topo_iteration else ''
            if len(name) == 0:
                path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images_topo' + iter + '/epoch_' + str(
                    epoch) + '/'
            else:
                path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images_topo' + iter + '/epoch_' + str(
                    epoch) + '/' + name[0] + '/'
        else:
            if len(name) == 0:
                path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images' + '/epoch_' + str(epoch) + '/'
            else:
                path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images' + '/epoch_' + str(epoch) + '/' + name[0] + '/'

        if not os.path.exists(path):
            os.makedirs(path)
        # SAVE Valid Likelihood Images and Prediction
        export_name_lh = str(batch) + '-{}lh.png'.format(i)
        export_name_pred = str(batch) + '-{}pred.png'.format(i)
        img.save(path + export_name_lh)
        pred.save(path + export_name_pred)


def save_groundTrue_att(images, labels, args, batch, epoch):
    mid = int(args.step_size/2)
    img_as_np = images[:, mid].cpu().data.numpy()
    label_as_np = labels[:, mid].cpu().data.numpy()

    img_as_np, label_as_np = img_as_np * 255, label_as_np * 255
    img_as_np, label_as_np = img_as_np.astype(np.uint8), label_as_np.astype(np.uint8)
    # print(img_as_np, img_as_np.shape)

    img, label = Image.fromarray(img_as_np.squeeze(0)), Image.fromarray(label_as_np.squeeze(0))
    if args.topo_attention:
        iter = '_iter' if args.topo_iteration else ''
        path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images_topo' + iter + '/epoch_' + str(
            epoch) + '/'
    else:
        path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images' + '/epoch_' + str(epoch) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    # SAVE Valid ground true Images
    export_name_orig = str(batch) + 'orig.png'
    export_name_gt = str(batch) + 'gt.png'
    img.save(path + export_name_orig)
    label.save(path + export_name_gt)


def save_groundTrue(images, labels, args, batch, epoch, name):
    for i in range(images.shape[1]):
        img_as_np = images[:,i].cpu().data.numpy()
        label_as_np = labels[:,i].cpu().data.numpy()

        img_as_np, label_as_np = img_as_np * 255, label_as_np * 255
        img_as_np, label_as_np = img_as_np.astype(np.uint8), label_as_np.astype(np.uint8)
        # print(img_as_np, img_as_np.shape)

        img, label = Image.fromarray(img_as_np.squeeze(0)), Image.fromarray(label_as_np.squeeze(0))
        if args.topo_attention:
            iter = '_iter' if args.topo_iteration else ''
            if len(name) == 0:
                path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images_topo' + iter + '/epoch_' + str(
                    epoch) + '/'
            else:
                path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images_topo' + iter + '/epoch_' + str(
                    epoch) + '/'  + name[0] + '/'
        else:
            if len(name) == 0:
                path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images' + '/epoch_' + str(epoch) + '/'
            else:
                path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images' + '/epoch_' + str(epoch) + '/'  + name[0] + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        # SAVE Valid ground true Images
        export_name_orig = str(batch) + '-{}orig.png'.format(i)
        export_name_gt = str(batch) + '-{}gt.png'.format(i)
        img.save(path + export_name_orig)
        label.save(path + export_name_gt)



def save_attention_features(cp_group, batch, epoch, args):
#    img_cp0 = Image.fromarray((cp0.cpu().numpy() * 255).astype(np.uint8))
#    img_cp1 = Image.fromarray((cp1.cpu().numpy() * 255).astype(np.uint8))
#    img_cp2 = Image.fromarray((cp2.cpu().numpy() * 255).astype(np.uint8))
    iter = '_iter' if args.topo_iteration else ''
    path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images_topo' + iter + '/epoch_' + str(epoch) + '/' + str(batch) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    for i, cp in enumerate(cp_group):
        img_cp = Image.fromarray((cp.cpu().numpy() * 255).astype(np.uint8))
        export_name_cp = 'cp{}.png'.format(i)
        img_cp.save(path + export_name_cp)
#    export_name_cp0 = 'cp0.png'
#    export_name_cp1 = 'cp1.png'
#    export_name_cp2 = 'cp2.png'

#    img_cp0.save(path + export_name_cp0)
#    img_cp1.save(path + export_name_cp1)
#    img_cp2.save(path + export_name_cp2)

def save_attention_score(out, batch, epoch, args, i):
    img = Image.fromarray((out.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8))
    iter = '_iter' if args.topo_iteration else ''
    path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images_topo' + iter + '/epoch_' + str(epoch) + '/' + str(batch) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    export_name = 'score{0}.png'.format(i)
    img.save(path + export_name)

def save_attention(attention, batch, epoch, args):
    iter = '_iter' if args.topo_iteration else ''
    path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images_topo' + iter + '/epoch_' + str(epoch) + '/' + str(batch) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    img = Image.fromarray((attention.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8))
    export_name = 'attention.png'
    img.save(path + export_name)


def save_likelihood(likelihood, batch, epoch, args):
    inLikelihood = Image.fromarray((likelihood[0].squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8))
    outLikelihood = Image.fromarray((likelihood[1].squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8))
    iter = '_iter' if args.topo_iteration else ''
    path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images_topo'  + iter + '/epoch_' + str(epoch) + '/' + str(batch) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    export_name_in = 'lh_in.png'
    export_name_out = 'lh_out.png'

    inLikelihood.save(path + export_name_in)
    outLikelihood.save(path + export_name_out)
