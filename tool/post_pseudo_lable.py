import os 
import shutil
import tqdm
import glob

save_dir  = "../DATA/license_pseudo"

pseudo_label_path = '../license_pseudo_2.txt'
prob_split = {
    'remove': [0, 0.0001],
    'super_weak': [0.0001, 0.1],
    'weak': [0.1, 0.3],
    'medium': [0.3, 0.5],
    'high': [0.5, 1]
}
sep = '||||'

folders = prob_split.keys()
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(save_dir + "/" + folder, exist_ok=True)

files = {
    'remove': open(save_dir + "/" + 'remove.txt', 'w'),
    'super_weak': open(save_dir + "/" + 'super_weak.txt', 'w'), 
    'weak': open(save_dir + "/" + 'weak.txt', 'w'),
    'medium': open(save_dir + "/" + 'medium.txt', 'w'),
    'high': open(save_dir + "/" + 'high.txt', 'w')
}

def process(write_txt=False)):
    with open(pseudo_label_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    for line in tqdm.tqdm(lines):
        img_path, label, prob = line.split(sep)
        prob = float(prob)
        print(prob)
        for key in prob_split.keys():
            print(prob_split[key])
            if prob_split[key][0] <= prob < prob_split[key][1]:
                shutil.copy("../" + img_path, os.path.join(save_dir, key))
                files[key].write(os.path.join(key, os.path.basename(img_path)) + "||||" + label + "\n")
                break

    for file in files.values():
        file.close()

def re_label(img_dir, label_file):
    """
    remove images in label that not include in img_dir
    """
    paths = os.listdir(img_dir)
    with open(label_file, 'r') as f, open("re_" + label_file, 'w') as f_w:
        for line in f:
            imgpath, _ = line.split("||||")
            if os.path.basename(imgpath) in paths:
                f_w.write(line)


def label2txt(label_file, result_dir):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    for line in lines:
        imgpath, label = line.split("||||")
        shutil.copy(imgpath, result_dir)
        with open(os.path.join(result_dir, os.path.basename(imgpath).split(".")[0] + ".txt"), 'w', encoding='utf-8') as f:
            f.write(label) 



def txt2label(data_dir, label_file, remove_txt=False):
    txt_files = glob.glob(data_dir + "/*.txt")
    img_files = glob.glob(data_dir + "/*.jpg")
    # assert len(txt_files) == len(img)
    with open(label_file, 'w', encoding='utf8') as f_w: 
        for txt in txt_files:
            if os.path.splitext(txt)[0] + ".jpg" not in img_files:
                if remove_txt:
                    os.remove(txt)
                continue
            with open(txt ,'r', encoding='utf8') as f:
                label = f.readline().strip()
                line_new = data_dir + os.path.splitext(txt)[0] + ".jpg" +"||||" + label + "\n"
                f_w.write(line_new)
            if remove_txt:
                os.remove(txt)

                
if __name__ == "__main__":
    # process()
    # label2txt("remove.txt", 'remove_ed')
    txt2label(data_dir='remove_ed', label_file='remove_ed.txt', remove_txt=False)