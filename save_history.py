import csv
import os

import torch


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
                'optimizer' : optimizer.state_dict(),
            }, path + "/model_epoch_{0}.pwf".format(epoch))
