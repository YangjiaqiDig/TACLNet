# from topoLoss import *
import logging
import time
from PIL import Image
from TDFMain_pytorch import *
from betti_compute import betti_number
from pre_processing import *
import torch
from save_history import *

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
    # print('bcp_lh.shape', bcp_lh.shape)
    for coor in bcp_lh:
        if any(coor < 0) or any(coor >= criticPointMap.shape[0]) or any(coor >= criticPointMap.shape[1]): continue
        criticPointMap[int(coor[0])][int(coor[1])] = 1

    criticPointMap = addGaussianFilter(criticPointMap, args)
    criticPointMap = criticPointMap.squeeze(0).squeeze(0)
    criticPointMap = (criticPointMap - torch.min(criticPointMap)) * 1 / (torch.max(criticPointMap) - torch.min(criticPointMap))

    return criticPointMap

def get_critical_points(likelihoodMap, label, groundtruth, args):
    criticPointMap = torch.zeros(likelihoodMap.shape).to(args.device)
    pd_lh, bcp_lh, dcp_lh, lh_pers, lh_pers_valid = getPers(likelihoodMap)
    criticPointMap = getTopoFilter(criticPointMap, bcp_lh, args)
    return criticPointMap


def get_critical_points_patch(likelihoodMap, label, predict, args):
    topo_size = args.topo_size
    gt_dmap = label.to(args.device)
    et_dmap = likelihoodMap
    criticPointMapAll = torch.zeros(likelihoodMap.shape).to(args.device)
    for y in range(0, et_dmap.shape[0], topo_size):
        for x in range(0, et_dmap.shape[1], topo_size):
            likelihood = et_dmap[y:min(y + topo_size, et_dmap.shape[0]),
                         x:min(x + topo_size, et_dmap.shape[1])]
            groundtruth = gt_dmap[y:min(y + topo_size, et_dmap.shape[0]),
                          x:min(x + topo_size, et_dmap.shape[1])]
            binary = predict[y:min(y + topo_size, et_dmap.shape[0]),
                     x:min(x + topo_size, et_dmap.shape[1])]
            criticPointMap = torch.zeros(likelihood.shape).to(args.device)
            # print('likelihood', likelihood, 'groundtruth', groundtruth, 'binaryPredict', binary)
            predict_betti_number = betti_number(binary)
            groundtruth_betti_number = betti_number(groundtruth)

            if (torch.min(likelihood) == 1 or torch.max(likelihood) == 0):
                continue
            if (torch.min(groundtruth) == 1 or torch.max(groundtruth) == 0):
                continue
            if groundtruth_betti_number == 0:
                continue
            if (abs(predict_betti_number - groundtruth_betti_number) / groundtruth_betti_number) < 0.7:
                continue
            if (len(likelihood.shape) < 2 or len(groundtruth.shape) < 2):
                continue
            print('row: ', y, 'col: ', x)
            pd_lh, bcp_lh, dcp_lh, lh_pers, lh_pers_valid = getPers(likelihood)
            criticPointMap = getTopoFilter(criticPointMap, bcp_lh, args)
            # saveForTest(criticPointMap, i, type='cp')
            # saveForTest(likelihood, i , type='lh')
            criticPointMapAll[y:min(y + topo_size, et_dmap.shape[0]),
            x:min(x + topo_size, et_dmap.shape[1])] = criticPointMap
    return criticPointMapAll


def saveForTest(img, i, type):
    img[img > 1] = 1
    img[img < 0] = 0
    # cp = 1 - cp
    # print(img)
    img = img * 255
    img = img.detach().cpu().numpy().astype(np.uint8)
    # print(img)
    img = Image.fromarray(img)
    img.save('cpCheck/' + str(i) + type + '.png')


def convert_topo(likelihoodMaps, labels, predict_all, args):
    dataSet = []
    for i, likelihoodMap in enumerate(likelihoodMaps):
        cp = get_critical_points(likelihoodMap, labels[i], predict_all[i], args)
        newData = torch.cat((cp.unsqueeze(0), likelihoodMap.unsqueeze(0)), dim=0)
        dataSet.append(newData)

    dataSet = torch.stack(dataSet)  # (n, 2, size, size)
    train = (dataSet, labels)
    torch.save(train, args.topo_dataset_cache)

    return train


def downsampling(likelihoodMap, times=2):
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    for i in range(times):
        likelihoodMap = maxpool(likelihoodMap)

    return likelihoodMap

def upsampling(likelihoodMap, times=2):
    up = [nn.Upsample(scale_factor=(2, 2), mode="nearest"), nn.Upsample(size=(625, 625), mode="nearest")]

    for i in range(times):
        likelihoodMap = up[i](likelihoodMap)
        # print(likelihoodMap.shape)
    return likelihoodMap

def topo_attention(output, labels, args, batch=0, epoch=0, valid=False):

    start = time.time()
    output = 1 - output
    labels = 1- labels
    softmax = torch.nn.Softmax2d()

    # [batch, 6, size, size], [batch, 3, size, size]
    V_lh = softmax(output[:, 2:4])[:, 1]
    output = downsampling(output)
    labels = downsampling(labels.float())

    q_lh = softmax(output[:, :2])[:, 1]
    v_lh = softmax(output[:, 2:4])[:, 1]  # [batch, size, size]
    k_lh = softmax(output[:, -2:])[:, 1]

    q_pred = q_lh >= 0.5
    v_pred = v_lh >= 0.5
    k_pred = k_lh >= 0.5

    q, v, k = [], [], []

    for i in range(len(v_lh)):
        q_cp = get_critical_points_patch(q_lh[i], labels[i][0], q_pred[i], args)  # (size, size)
        v_cp = get_critical_points_patch(v_lh[i], labels[i][1], v_pred[i], args)  # (size, size)
        k_cp = get_critical_points_patch(k_lh[i], labels[i][2], k_pred[i], args)  # (size, size)

        q.append(q_cp)
        v.append(v_cp)
        k.append(k_cp)

        if valid:
            save_attention_features(q_cp, v_cp, k_cp, batch, epoch, args)

    q, v, k = torch.stack(q, 0), torch.stack(v, 0), torch.stack(k, 0)

    # print(q.shape, v.shape, k.shape)  # （batch, size, size)
    batchSize = q.shape[0]
    w, h = q.shape[1], q.shape[2]
    proj_query = q.view(batchSize, -1, w*h).permute(0,2,1).cpu()
    proj_key = k.view(batchSize, -1, w*h).cpu()

    energy = torch.bmm(proj_query, proj_key)
    softmax1D = torch.nn.Softmax(dim=-1)
    attention = softmax1D(energy)

    proj_value = v.view(batchSize, -1, w*h).cpu()
    out = torch.bmm(proj_value, attention.permute(0, 2, 1))
    out = out.view(batchSize, w, h)
    """Normalize Attention Score"""
    for i in range(len(out)):
        out[i] = (out[i] - torch.min(out[i]))/ (torch.max(out[i]) - torch.min(out[i]))
    if valid: save_attention_score(out, batch, epoch, args)

    att = torch.mul(out.to(args.device), v_lh)
    attention = upsampling(att.unsqueeze(1)).squeeze(1)
    if valid: save_attention(attention, batch, epoch, args)

    output = 0.5*attention + V_lh
    # for i in range(len(output)):
    #     output[i] = (output[i] - torch.min(output[i])) / (torch.max(output[i]) - torch.min(output[i]))
    output[output > 1] = 1
    output[output < 0] = 0
    if valid: save_likelihood([V_lh, output], batch, epoch, args)
    print('running time: ', time.time() - start)
    output = torch.stack((output, (1-output)), dim=1)

    return output


