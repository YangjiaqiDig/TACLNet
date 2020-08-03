import logging
import os
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


def get_dataset_lstm(args):
    originalData = load_preprocess_dataset(args)
    originalDataSet = DataLoaderForUnet(originalData)
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
            predict = likelihoodMap >= 0.5
            likelihood_map_all.append(likelihoodMap)
            predict_all.append(predict)
    likelihood_map_all = torch.cat(likelihood_map_all, dim=0)  # (n, size, size)
    predict_all = torch.cat(predict_all, dim=0)
    # print(likelihood_map_all[0], originalData[1].shape, originalData[1][0])
    # TODO: save and load this dataset likelihood for certain round of epoch trained model.
    train = convert_topo(likelihood_map_all, originalData[1], predict_all, args)

    trainDataSet = DataLoaderForUnet(train)
    logging.info("DataSet for CLSTM shape %s", train[0].shape)
    return trainDataSet

def get_dataset(args):
    train = load_preprocess_dataset(args)
    validSep_beg = int(len(train[0]) / 5 * (args.valid_round - 1))
    validSep_end = int(len(train[0]) / 5 * args.valid_round)

    trainData = (torch.cat([train[0][:validSep_beg], train[0][validSep_end:]], dim=0),
                 torch.cat([train[1][:validSep_beg], train[1][validSep_end:]], dim=0))
    validData = (train[0][validSep_beg:validSep_end],
                 train[1][validSep_beg:validSep_end])
    trainDataSet = DataLoaderForUnet(trainData)
    validDataSet = DataLoaderForUnet(validData)
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


def save_prediction(likelihood_map, pred_class, args, batch, epoch):
    img_as_np, pred_as_np = likelihood_map.cpu().data.numpy(), pred_class.cpu().data.numpy()

    img_as_np, pred_as_np = img_as_np * 255, pred_as_np * 255
    img_as_np, pred_as_np = img_as_np.astype(np.uint8), pred_as_np.astype(np.uint8)
    # print(img_as_np, img_as_np.shape)
    img, pred = Image.fromarray(img_as_np.squeeze(0)), Image.fromarray(pred_as_np.squeeze(0))
    path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images' + '/epoch_' + str(epoch) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    # SAVE Valid Likelihood Images and Prediction
    export_name_lh = str(batch) + 'lh.png'
    export_name_pred = str(batch) + 'pred.png'
    img.save(path + export_name_lh)
    pred.save(path + export_name_pred)


def save_groundTrue(images, labels, args, batch, epoch):
    img_as_np = images.cpu().data.numpy()
    label_as_np = labels.cpu().data.numpy()

    img_as_np, label_as_np = img_as_np * 255, label_as_np * 255
    img_as_np, label_as_np = img_as_np.astype(np.uint8), label_as_np.astype(np.uint8)
    # print(img_as_np, img_as_np.shape)

    img, label = Image.fromarray(img_as_np.squeeze(0)), Image.fromarray(label_as_np.squeeze(0))
    path = args.save_folder + '/valid_' + str(args.valid_round) + '/saved_images' + '/epoch_' + str(epoch) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    # SAVE Valid ground true Images
    export_name_orig = str(batch) + 'orig.png'
    export_name_gt = str(batch) + 'gt.png'
    img.save(path + export_name_orig)
    label.save(path + export_name_gt)


if __name__ == "__main__":
    x = torch.tensor([[[1, 2]], [[3, 4]], [[3, 4]]])
    y = torch.tensor([[[1, 2]], [[3, 4]], [[3, 4]]])
    z = [x, y]
    print(x.shape, y.shape)
    q = torch.cat(z, dim=0)
    print(q.shape)
    print(q.squeeze(dim=1).shape)
