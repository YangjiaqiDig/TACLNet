# from topoLoss import *
import logging
import time

from TDFMain_pytorch import *
from betti_compute import betti_number
from pre_processing import *
import torch
# logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)

torch.set_printoptions(edgeitems=60)
np.set_printoptions(edgeitems=60)

n_fix = 0
n_remove = 0
pers_thd_lh = 0.03
pers_thd_gt = 0.03


def getPers(likelihood):
    pd_lh, bcp_lh, dcp_lh = compute_persistence_2DImg_1DHom_lh(likelihood)

    if (pd_lh.shape[0] > 0):
        lh_pers = pd_lh[:, 1] - pd_lh[:, 0]
        lh_pers_valid = lh_pers[np.where(lh_pers > pers_thd_lh)]
    else:
        lh_pers = np.array([])
        lh_pers_valid = np.array([])

    return pd_lh, bcp_lh, dcp_lh, lh_pers, lh_pers_valid


def getTopoFilter(criticPointMap, bcp_lh, args):
    # print('criticPointMap', criticPointMap.shape)
    # print('bcp_lh.shape', bcp_lh.shape)
    for coor in bcp_lh:
        if any(coor < 0) or any(coor >= criticPointMap.shape[0]): continue
        criticPointMap[int(coor[0])][int(coor[1])] = 1

    criticPointMap = addGaussianFilter(criticPointMap, args)
    criticPointMap = criticPointMap.squeeze(0).squeeze(0)

    return criticPointMap


def get_critical_points(likelihoodMap, label, predict, args):
    topo_size = args.topo_size
    gt_dmap = label.to(args.device)
    et_dmap = likelihoodMap
    start = time.time()
    criticPointMapCollect = []
    i=0
    for y in range(0, et_dmap.shape[0], topo_size):
        for x in range(0, et_dmap.shape[1], topo_size):
            criticPointMap = torch.zeros(topo_size, topo_size).to(args.device)
            likelihood = et_dmap[y:min(y + topo_size, et_dmap.shape[0]),
                         x:min(x + topo_size, et_dmap.shape[1])]
            groundtruth = gt_dmap[y:min(y + topo_size, et_dmap.shape[0]),
                          x:min(x + topo_size, et_dmap.shape[1])]
            binary = predict[y:min(y + topo_size, et_dmap.shape[0]),
                     x:min(x + topo_size, et_dmap.shape[1])]

            # print('likelihood', likelihood, 'groundtruth', groundtruth, 'binaryPredict', binary)
            predict_betti_number = betti_number(binary)
            groundtruth_betti_number = betti_number(groundtruth)

            if (torch.min(likelihood) == 1 or torch.max(likelihood) == 0):
                criticPointMapCollect.append(criticPointMap)
                continue
            if (torch.min(groundtruth) == 1 or torch.max(groundtruth) == 0):
                criticPointMapCollect.append(criticPointMap)
                continue
            if groundtruth_betti_number == 0:
                criticPointMapCollect.append(criticPointMap)
                continue
            if (abs(predict_betti_number - groundtruth_betti_number) / groundtruth_betti_number) < 0.4:
                criticPointMapCollect.append(criticPointMap)
                continue
            if (len(likelihood.shape) < 2 or len(groundtruth.shape) < 2):
                criticPointMapCollect.append(criticPointMap)
                continue
            print('row: ', y, 'col: ', x)
            pd_lh, bcp_lh, dcp_lh, lh_pers, lh_pers_valid = getPers(likelihood)
            criticPointMap = getTopoFilter(criticPointMap, bcp_lh, args)
            # saveForTest(criticPointMap, i, type='cp')
            # saveForTest(likelihood, i , type='lh')
            criticPointMapCollect.append(criticPointMap)
            i+=1
    print('running time: ', time.time()-start)
    criticPointMapCollect = torch.stack(criticPointMapCollect)
    criticPointMapCollect = criticPointMapCollect.view(likelihoodMap.shape)

    return criticPointMapCollect

def saveForTest(img, i, type):
    img[img > 1] = 1
    img[img < 0] = 0
    # cp = 1 - cp
    # print(img)
    img = img * 255
    img = img.cpu().numpy().astype(np.uint8)
    # print(img)
    img = Image.fromarray(img)
    img.save('cpCheck/' + str(i) + type + '.png')

def convert_topo(likelihoodMaps, labels, predict_all, args):
    dataSet = []
    start = time.time()
    for i, likelihoodMap in enumerate(likelihoodMaps):
        cp = get_critical_points(likelihoodMap, labels[i], predict_all[i], args)
        newData = torch.cat((cp.unsqueeze(0), likelihoodMap.unsqueeze(0)), dim=0)
        dataSet.append(newData)

    dataSet = torch.stack(dataSet) # (n, 2, size, size)
    print(dataSet.shape, 'convert to topo running time: ', time.time() - start)
    train = (dataSet, labels)
    torch.save(train, args.topo_dataset_cache)

    return train

def loss_topo_attention(output, labels, args):
    softmax = torch.nn.Softmax2d()

    likelihoodMaps = softmax(output)[:, 1, :, :] # (batch, size, size)
    predict_all = torch.argmax(output, dim=1) # (batch, size, size)

    attention = []
    gt = []
    for i, likelihoodMap in enumerate(likelihoodMaps):
        cp = get_critical_points(likelihoodMap, labels[i], predict_all[i], args) # (size, size)
        topoAttention = torch.mm(cp, likelihoodMap)
        groundTrue = torch.mm(cp, labels[i])
        attention.append(topoAttention)
        gt.append(groundTrue)
    attention = torch.stack(attention, 0)
    gt = torch.stack(gt, 0)

    return attention, gt





