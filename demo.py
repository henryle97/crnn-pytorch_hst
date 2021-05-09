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
import tqdm

BATCH_SIZE = 128
class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.paths = glob.glob(folder + "/*.jpg")
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        if self.transform is not None:
            img = self.transform(img)

        return img, self.paths[idx]

    def __len__(self):
        return len(self.paths)

def get_data(folder):
    transformer = dataset.processing_image((params.imgW, params.imgH))
    folder_dataset = FolderDataset(folder, transform=transformer)
    folder_dataloader = torch.utils.data.DataLoader(folder_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return folder_dataloader

def testing(model, input_dir=None, pseudo_label=None):
    image_paths = glob.glob(input_dir + "/*.jpg")
    converter = utils.strLabelConverter(params.alphabet)
    transformer = dataset.processing_image((params.imgW, params.imgH))
    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert('L')
            image = transformer(image)
            if args.check: 
                cv2.imwrite('DATA/img_check/' + os.path.basename(image_path) + ".jpg", image.mul_(0.5).add_(0.5).permute(1, 2, 0).numpy()* 255.0)

            if torch.cuda.is_available():
                image = image.cuda()
            image = image.view(1, *image.size())
            image = Variable(image)

            preds = model(image)     # SxBxC
            probs = torch.exp(preds)
            max_probs, preds = probs.max(2)
            prob = max_probs.cumprod(0)[-1]
            # from IPython import embed; embed()

            # _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)

            preds_size = Variable(torch.LongTensor([preds.size(0)]))
            raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
            sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
            print('Text in %-65s is:  %-12s - prob: %-6f' % (image_path, sim_pred, prob))


def pseudo_label(model, input_dir, output_file=None):
    image_paths = glob.glob(input_dir + "/*.jpg")
    converter = utils.strLabelConverter(params.alphabet)
    transformer = dataset.processing_image((params.imgW, params.imgH))
    with torch.no_grad():
        with open(output_file, 'w') as f:
            for image_path in tqdm.tqdm(image_paths):
                image = Image.open(image_path).convert('L')
                image = transformer(image)
                if args.check: 
                    cv2.imwrite('DATA/img_check/' + os.path.basename(image_path) + ".jpg", image.mul_(0.5).add_(0.5).permute(1, 2, 0).numpy()* 255.0)

                if torch.cuda.is_available():
                    image = image.cuda()
                image = image.view(1, *image.size())
                image = Variable(image)

                preds = model(image)     # SxBxC
                preds_exp = torch.exp(preds)
                max_probs, preds = preds_exp.max(2)
                prob = max_probs.cumprod(0)[-1]

                # _, preds = preds.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)

                preds_size = Variable(torch.LongTensor([preds.size(0)] * BATCH_SIZE))
                # raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
                sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
                for pred, prob in zip(sim_pred, probs):
                # print('Text in %-65s is:  %-12s - prob: %-6f' % (image_path, sim_pred, prob))
                    info = "||||".join([image_path, pred, str(float(probs))]) + "\n"
                    f.write(info)


def pseudo_label_batch(model, input_dir, output_file=None):
    dataloader_ = get_data(input_dir)
    converter = utils.strLabelConverter(params.alphabet)
    with torch.no_grad():
        with open(output_file, 'w') as f:
            for img_batch, img_paths in tqdm.tqdm(dataloader_):
                if torch.cuda.is_available():
                    img_batch = img_batch.cuda()

                preds = model(img_batch)     # SxBxC
                preds_size = Variable(torch.LongTensor([preds.size(0)] * preds.size(1)))  # (B,)
                preds_exp = torch.exp(preds)

                max_probs, preds = preds_exp.max(2)           
                probs = max_probs.cumprod(0)[-1].cpu().numpy()   # (B,)
                preds = preds.transpose(1, 0).contiguous().view(-1)  

                sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
                
                for idx, (pred, prob) in enumerate(zip(sim_preds, probs)):
                    info = "||||".join([img_paths[idx], pred, str(float(prob))]) + "\n"
                    f.write(info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type = str, required = True, help = 'crnn model path')
    parser.add_argument('-i', '--image_dir', type = str, required = True, help = 'demo image path')
    parser.add_argument('-check', action='store_true', default=False, help = 'demo image path')

    args = parser.parse_args()

    model_path = args.model_path
    
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

    testing(model, input_dir=args.image_dir)
    # pseudo_label_batch(model, input_dir=args.image_dir, output_file='license_pseudo_2.txt')