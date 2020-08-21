import os

import numpy as np
from PIL import Image
from dataset import *
from model import UNET
from topo import *

# logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)


def load_preprocess_dataset(args):
    train_path = args.dataset_path_train
    label_path = args.dataset_path_label
    if args.dataset_cache and os.path.isfile(args.dataset_cache):
        logging.info("Load enhanced dataset before DataLoader from cache at %s", args.dataset_cache)
        train = torch.load(args.dataset_cache)

    else:
        logging.info("Start Prepare enhanced dataset before DataLoader %s", train_path)
        train = DataTrain(train_path, label_path)
        torch.save(train, args.dataset_cache)

    return train


def get_dataset_clstm(args):
    if args.topo_dataset_cache and os.path.isfile(args.topo_dataset_cache):
        logging.info("Load critical points dataset before CLTM DataLoader from cache at %s", args.dataset_cache)
        train_with_cp = torch.load(args.topo_dataset_cache)
    else:
        logging.info("Start Prepare critical points dataset before CLTM DataLoader")
        originalData = load_preprocess_dataset(args)  # label (n, size, size)

        # print(msk_as_np.shape)
        originalDataSet = UNETDataSet(originalData)
        origin_loader = torch.utils.data.DataLoader(dataset=originalDataSet, num_workers=6,
                                                    batch_size=args.train_batch_size,
                                                    shuffle=False)
        model = UNET()
        if args.device == "cuda":
            model = torch.nn.DataParallel(model, device_ids=list(
                range(torch.cuda.device_count()))).cuda()
        path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_models' + args.check_point
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        likelihood_map_all = []
        predict_all = []
        for batch, data in enumerate(origin_loader):
            images, labels = data[0].unsqueeze(1), data[1]

            with torch.no_grad():
                output, likelihoodMap = model(images.to(args.device))  # (batch, 1, size, size)
                # likelihoodMap = downsampling(likelihoodMap, times=2)
                predict = likelihoodMap >= 0.5
                likelihood_map_all.append(likelihoodMap)
                predict_all.append(predict)

        likelihood_map_all = torch.cat(likelihood_map_all, dim=0)  # (n, size, size)
        predict_all = torch.cat(predict_all, dim=0)
        label_all = originalData[1]
        # label_all = downsampling(label_all.unsqueeze(1)).squeeze(1)
        # print(label_all.shape)

        # print(likelihood_map_all[0], originalData[1].shape, originalData[1][0])
        # image = (down2.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        # image = Image.fromarray(image)
        # image.save('downlh.png')
        # TODO: save and load this dataset likelihood for certain round of epoch trained model.
        train_with_cp = convert_topo(likelihood_map_all, label_all, predict_all, args)  # (n, 2, size, size)
    logging.info("DataSet for CLSTM shape %s", train_with_cp[0].shape)
    train_with_cp = list(train_with_cp)
    train_with_cp[0] = train_with_cp[0].cpu()
    coppedTrain = train_with_cp[0]  # chunkCrop(train_with_cp[0])
    coppedLabel = train_with_cp[1]  # chunkCrop(train_with_cp[1])

    trainDataSet = LSTMDataSet([coppedTrain, coppedLabel], args)  # (n, 3, 2, size, size)

    return trainDataSet



def get_dataset_topoClstm(args):
    train = load_preprocess_dataset(args)
    expandChannelDimTrain = train[0].unsqueeze(1)

    seq_train, seq_label = lstmDataTrain([expandChannelDimTrain, train[1]], args)
    dataset = UNETDataSet([seq_train, seq_label])

    validation_split = 1 / 3

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    splitIndex = split * (args.valid_round - 1)
    train_indices, val_indices = indices[:splitIndex] + indices[splitIndex + split:], indices[
                                                                                      splitIndex: splitIndex + split]
    # print(train_indices, val_indices)

    train_sampler = torch.utils.data.Subset(dataset, train_indices)
    valid_sampler = torch.utils.data.Subset(dataset, val_indices)

    return train_sampler, valid_sampler


def get_dataset(args):
    train = load_preprocess_dataset(args)
    validSep_beg = int(len(train[0]) / 5 * (args.valid_round - 1))
    validSep_end = int(len(train[0]) / 5 * args.valid_round)

    trainData = (torch.cat([train[0][:validSep_beg], train[0][validSep_end:]], dim=0),
                 torch.cat([train[1][:validSep_beg], train[1][validSep_end:]], dim=0))
    validData = (train[0][validSep_beg:validSep_end],
                 train[1][validSep_beg:validSep_end])
    trainDataSet = UNETDataSet(trainData)
    validDataSet = UNETDataSet(validData)
    logging.info("TrainDataSet shape %s", trainData[0].shape)

    return (trainDataSet, validDataSet)


def accuracy_check(labels, pred_class):
    ims = [labels, pred_class]
    np_ims = []
    for item in ims:
        if 'str' in str(type(item)):
            item = np.array(Image.open(item))
        elif 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.numpy()
        np_ims.append(item)

    compare = np.equal(np_ims[0], np_ims[1])
    accuracy = np.sum(compare)

    return accuracy / len(np_ims[0].flatten())


def accuracy_for_batch(labels, pred_class, args):
    total_acc = 0
    batch_size = labels.size()[0]
    for index in range(batch_size):
        total_acc += accuracy_check(labels, pred_class)
    return total_acc / batch_size


if __name__ == "__main__":
    x = torch.tensor([[[1, 2]], [[3, 4]], [[3, 4]]])
    y = torch.tensor([[[1, 2]], [[3, 4]], [[3, 4]]])
    z = [x, y]
    print(x.shape, y.shape)
    q = torch.cat(z, dim=0)
    print(q.shape)
    print(q.squeeze(dim=1).shape)
