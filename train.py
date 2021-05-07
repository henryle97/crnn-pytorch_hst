from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss
import os
import utils
import dataset

import models.crnn as net
import params
from aug import ImgAugTransform
import cv2
import math
import matplotlib.pyplot as plt
from tool.logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument('-train', '--trainroot', required=True, help='path to train dataset')
parser.add_argument('-val', '--valroot', required=True, help='path to val dataset')
parser.add_argument('-error', action='store_true', default=False)
parser.add_argument('-vis_data', action='store_true', default=False)
args = parser.parse_args()

if not os.path.exists(params.expr_dir):
    os.makedirs(params.expr_dir)

# ensure everytime the random is the same
random.seed(params.manualSeed)
np.random.seed(params.manualSeed)
torch.manual_seed(params.manualSeed)

cudnn.benchmark = True

# Logger
logger = Logger(log_dir=params.expr_dir)

if torch.cuda.is_available() and not params.cuda:
    print("WARNING: You have a CUDA device, so you should probably set cuda in params.py to True")

# -----------------------------------------------
"""
In this block
    Get train and val data_loader
"""
def data_loader():
    # train
    train_transform = ImgAugTransform()
    train_dataset = dataset.lmdbDataset(root=args.trainroot, transform=train_transform)
    assert train_dataset
    if not params.random_sample:
        sampler = dataset.randomSequentialSampler(train_dataset, params.batchSize)
    else:
        sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batchSize, \
            shuffle=True, sampler=sampler, num_workers=int(params.workers), \
            collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))
    
    # val
    val_dataset = dataset.lmdbDataset(root=args.valroot, transform=dataset.processing_image((params.imgW, params.imgH)))
    assert val_dataset
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=params.batchSize, num_workers=int(params.workers))
    
    return train_loader, val_loader, train_dataset, val_dataset

train_loader, val_loader, train_dataset, val_dataset = data_loader()
info = 'Num traing samples: %d \n Num valid samples: %d \n %s ' % (len(train_dataset) , len(val_dataset), '=' * 100)


# -----------------------------------------------
"""
In this block
    Net init
    Weight init
    Load pretrained model
"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def net_init():
    nclass = len(params.alphabet) + 1
    crnn = net.CRNN(params.imgH, params.nc, nclass, params.nh)
    crnn.apply(weights_init)
    if params.pretrained != '':
        print('loading pretrained model from %s' % params.pretrained)
        if params.multi_gpu:
            crnn = torch.nn.DataParallel(crnn)
        crnn.load_state_dict(torch.load(params.pretrained))
    
    return crnn

crnn = net_init()
logger.info(crnn)

# -----------------------------------------------
"""
In this block
    Init some utils defined in utils.py
"""
# Compute average for `torch.Variable` and `torch.Tensor`.
loss_avg = utils.averager()
best_acc = -1e5

# Convert between str and label.
converter = utils.strLabelConverter(params.alphabet)

# -----------------------------------------------
"""
In this block
    criterion define
"""
criterion = CTCLoss()

# -----------------------------------------------
"""
In this block
    Init some tensor
    Put tensor and net on cuda
    NOTE:
        image, text, length is used by both val and train
        becaues train and val will never use it at the same time.
"""
image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
text = torch.LongTensor(params.batchSize * 5)
length = torch.LongTensor(params.batchSize)

if params.cuda and torch.cuda.is_available():
    criterion = criterion.cuda()
    image = image.cuda()
    text = text.cuda()

    crnn = crnn.cuda()
    if params.multi_gpu:
        crnn = torch.nn.DataParallel(crnn, device_ids=range(params.ngpu))

image = Variable(image)
text = Variable(text)
length = Variable(length)

# -----------------------------------------------
"""
In this block
    Setup optimizer
"""
if params.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
elif params.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

# -----------------------------------------------
"""
In this block
    Dealwith lossnan
    NOTE:
        I use different way to dealwith loss nan according to the torch version. 
"""
if params.dealwith_lossnan:
    if torch.__version__ >= '1.1.0':
        """
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.
        Pytorch add this param after v1.1.0 
        """
        criterion = CTCLoss(zero_infinity = True)
    else:
        """
        only when
            torch.__version__ < '1.1.0'
        we use this way to change the inf to zero
        """
        crnn.register_backward_hook(crnn.backward_hook)

# -----------------------------------------------

def val(net, criterion, show_error=False, img_check_dir='DATA/img_check'):
    global best_acc
    if show_error and not os.path.exists(img_check_dir):
        os.makedirs(img_check_dir, exist_ok=True)

    print('Start val')
    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    val_iter = iter(val_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager() # The blobal loss_avg is used by train

    max_iter = len(val_loader)
    errors = []
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        # from IPython import embed; embed()
        preds_exp = preds.exp()  # log_softmax -> softmax : SxBxC
        max_probs, preds = preds_exp.max(2)
        probs = max_probs.cumprod(0)[-1].cpu().numpy()
        preds = preds.transpose(1, 0).contiguous().view(-1)  # 1D  # BXSXC
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        cpu_texts_decode = []
        for i in cpu_texts:
            cpu_texts_decode.append(i.decode('utf-8', 'strict'))
        for idx,  (pred, target) in enumerate(zip(sim_preds, cpu_texts_decode)):
            if pred == target:
                n_correct += 1
            else:
                if show_error:

                    errors.append([pred, target])
                    cv2.imwrite('DATA/img_check/' + str(idx) + "_" + str(target) + ".jpg", image[idx].cpu().mul_(0.5).add_(0.5).permute(1, 2, 0).numpy()* 255.0)

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_val_disp]
    for raw_pred, pred, gt, prob in zip(raw_preds, sim_preds, cpu_texts_decode, probs):
        info = '%-20s ==> %-10s || Label: %-10s || prob: %-6.4f' % (raw_pred, pred, gt, prob)
        logger.info(info)

    accuracy = n_correct / float(len(val_dataset))
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(crnn.state_dict(), os.path.join(params.expr_dir, 'best.pth'))
    info = '[Epoch %d/%d]: Val loss: %f, accuray: %f, best_acc: %-10f \n %s' % (epoch, params.nepoch, loss_avg.val(), accuracy, best_acc, '='*100)
    logger.info(info)

    if show_error:
        for error in errors:
            print('pred: %-20s  - gt: %-20s' % (error[0], error[1]))


def train(net, criterion, optimizer, train_iter):
    for p in crnn.parameters():
        p.requires_grad = True
    crnn.train()

    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)
    
    optimizer.zero_grad()
    preds = crnn(image)
    preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    # crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

def visual_data(is_train=True, sample=20):
    if is_train:
        data_vis = train_loader
    else:
        data_vis = val_loader

    ncols = 5
    nrows = int(math.ceil(sample / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(12, 12))

    num_plots = 0
    for idx, batch in enumerate(data_vis):
        for vis_idx in range(len(batch)):
            row = num_plots // ncols
            col = num_plots % ncols

            img = batch[0][vis_idx].numpy().transpose(1, 2, 0)
            sent = batch[1][vis_idx].decode('utf-8', 'strict')

            ax[row, col].imshow(img, cmap='gray')
            ax[row, col].set_title("Label: {: <2}".format(sent), fontsize=16, color='g')

            ax[row, col].get_xaxis().set_ticks([])
            ax[row, col].get_yaxis().set_ticks([])

            num_plots += 1
            if num_plots >= sample:
                plt.subplots_adjust()
                fig.savefig(params.expr_dir + '/vis_dataset.png')
                return



if __name__ == "__main__":
    
    if args.error:
        val(crnn, criterion, show_error=True)
    elif args.vis_data:
        visual_data(is_train=True)
    else:

        for epoch in range(params.nepoch):
            train_iter = iter(train_loader)
            i = 0
            while i < len(train_loader):
                cost = train(crnn, criterion, optimizer, train_iter)
                loss_avg.add(cost)
                i += 1

                if i % params.displayInterval == 0:
                    infor = '[Epoch %d/%d][Iter %d/%d] Loss: %f' % (epoch, params.nepoch, i, len(train_loader), loss_avg.val())
                    logger.info(infor)
                    loss_avg.reset()

                if i % params.valInterval == 0:
                    val(crnn, criterion)

                # do checkpointing
                if i % params.saveInterval == 0:
                    torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(params.expr_dir, epoch, i))

            torch.save(crnn.state_dict(), os.path.join(params.expr_dir, 'last.pth'))