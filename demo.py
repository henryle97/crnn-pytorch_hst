import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn
import params
import argparse
import glob
import os
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type = str, required = True, help = 'crnn model path')
parser.add_argument('-i', '--image_dir', type = str, required = True, help = 'demo image path')
args = parser.parse_args()

model_path = args.model_path
image_paths = glob.glob(args.image_dir + "/*.jpg")



# net init
nclass = len(params.alphabet) + 1
model = crnn.CRNN(params.imgH, params.nc, nclass, params.nh)
if torch.cuda.is_available():
    model = model.cuda()

# load model
print('loading pretrained model from %s' % model_path)
if params.multi_gpu:
    model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(model_path))
model.eval()

converter = utils.strLabelConverter(params.alphabet)

transformer = dataset.processing_image((params.imgW, params.imgH))
for image_path in image_paths:
    image = Image.open(image_path).convert('L')
    print(image.size)
    image = transformer(image)
    cv2.imwrite('DATA/img_check/' + os.path.basename(image_path) + ".jpg", image.mul_(0.5).add_(0.5).permute(1, 2, 0).numpy()* 255.0)

    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.LongTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('Text in %-80s is:  %-20s' % (image_path, sim_pred))
