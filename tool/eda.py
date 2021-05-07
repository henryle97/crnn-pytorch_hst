import lmdb 
from collections import Counter
import six
import sys
import matplotlib.pyplot as plt
import tqdm
import os
import pandas as pd
from PIL import Image

class EDA_OCR:
    def __init__(self, lmdb_path, max_H=32, save_dir='EDA'):
        """
        data: lmdb data
        """
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_dir = save_dir
        self.data = {'size': [], 'label': []}

        self.env = lmdb.open(
            lmdb_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        if not self.env:
            print("Cannot find lmdb data at {}".format(lmdb_path))
            sys.exit(0)
        
        print("Start collect infor data...")
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode('utf-8')))
            for index in tqdm.tqdm(range(nSamples)):
                img_key = 'image-%09d' % index
                imgbuf = txn.get(img_key.encode('utf-8'))
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                try:
                    img = Image.open(buf).convert("L")
                    w, h = img.size
                    
                except Exception as err :
                    print("Corrupted image for %d: %s" % (index, err))
                    continue
                
                label_key = 'label-%09d' % index 
                label = txn.get(label_key.encode('utf-8'))

                self.data['size'].append([w, h])
                self.data['label'].append(label)

        self.max_H = max_H




    def visual_data(self):
        # visual size 
        width_list = []
        for size in self.data['size']:
            new_w, new_h = self.resize_img(size, max_h=self.max_H)
            width_list.append(new_w)

        plt.hist(width_list, bins=20)
        plt.savefig(os.path.join(self.save_dir, 'size_eda.jpg'))
        df = pd.DataFrame({'width': width_list})
        print(df.describe())

        # visual chars
        char_counter = Counter()
        for label in self.data['label']:
           char_counter.update(list(label.decode('utf-8', 'strict')))
        
        self.plot_bar(char_counter, 'char_eda.jpg', is_sorted=True)
    

    def resize_img(self, size,max_h=32):
        w, h = size 
        new_w = int(max_h * float(w/h))
        return new_w, max_h


    def plot_bar(self, counter, img_name, is_sorted=True):
        df = pd.DataFrame({'Items': list(counter.keys()), 'Freq': list(counter.values())})
        if is_sorted:
            df = df.sort_values('Freq', ascending=False)
        
        fig = plt.figure(figsize=(20, 10))
        plt.bar('Items', 'Freq', data=df)
        plt.xlim(-1, len(df))
        plt.xlabel('Items', fontsize=15)
        plt.ylabel('Freq', fontsize=15)
        fig.savefig(self.save_dir + "/" + img_name)
        

if __name__ == '__main__':
    eda = EDA_OCR(lmdb_path='../DATA/license_simple_120k')
    eda.visual_data()




