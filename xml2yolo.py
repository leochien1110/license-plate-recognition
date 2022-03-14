# --------------------------------
# convert xml annotation to txt file
# output: image.jpg + label.txt
# --------------------------------

import os
import shutil
import argparse
from xml.etree.ElementTree import parse
from bs4 import BeautifulSoup

def get_parser():
    parser = argparse.ArgumentParser(description='my description')
    parser.add_argument('mode', type=str)
    return parser

def run_convert(all_classes, train_img, train_annotation, yolo_path, write_txt):
    curr_path = os.getcwd()
    data_cnt = 0

    for data_file in os.listdir(train_annotation):
        try:
            with open(os.path.join(train_annotation, data_file), 'r') as f:
                print('reading file...')
                soup = BeautifulSoup(f.read(), 'xml')
                img_name = soup.select_one('filename').text
                
                for size in soup.select('size'):
                    img_w = int(size.select_one('width').text)
                    img_h = int(size.select_one('height').text)

                img_info = []
                for obj in soup.select('object'):
                    xmin = int(obj.select_one('xmin').text)
                    xmax = int(obj.select_one('xmax').text)
                    ymin = int(obj.select_one('ymin').text)
                    ymax = int(obj.select_one('ymax').text)
                    objclass = all_classes.get(obj.select_one('name').text)

                    # normalize bndbox center position
                    x = (xmin + (xmax-xmin)/2) * 1.0 / img_w
                    y = (ymin + (ymax-ymin)/2) * 1.0 / img_h
                    w = (xmax-xmin) * 1.0 / img_w
                    h = (ymax-ymin) * 1.0 / img_h
                    img_info.append(' '.join([str(objclass), str(x), str(y), str(w), str(h)]))

                # copy images to yolo path and rename
                img_path = os.path.join(train_img, img_name)
                img_format = img_name.split('.')[1]     # jpg or png
                print(img_name)
                shutil.copyfile(img_path, yolo_path + str(data_cnt) + '.' + img_format)

                # create yolo bndbox txt
                with open(yolo_path + str(data_cnt) + '.txt', 'a+') as f:
                    f.write('\n'.join(img_info))

                # create train/val txt
                with open(write_txt, 'a') as f:
                    path = os.path.join(curr_path, yolo_path)
                    line_txt = [path + str(data_cnt) + '.' + img_format, '\n']
                    f.writelines(line_txt)

                data_cnt += 1

        except Exception as e:
            print(e)

    print('Data converted!')


if __name__ == '__main__':
    # specify which data to convert
    parser = get_parser()
    args = parser.parse_args()
    print('mode: ' + args.mode)

    if args.mode == "train":
        train_img           = "data/train/images"
        train_annotation    = "data/train/annotations"
        yolo_path = "yolo_train/"
        write_txt = 'cfg/train.txt'
    elif args.mode == "val":
        train_img           = "data/val/images"
        train_annotation    = "data/val/annotations"
        yolo_path = "yolo_val/"
        write_txt = 'cfg/val.txt'
    
    all_classes = {'class_2':2, 'class_1':1, 'licence':0}   # class table(name, index)

    # Check directory, clean build
    if not os.path.exists(yolo_path):
        os.mkdir(yolo_path)
    else:
        lsdir = os.listdir(yolo_path)
        for name in lsdir:
            if name.endswith('.txt') or name.endswith('.jpg') or name.endswith('.png'):
                os.remove(os.path.join(yolo_path, name))

    cfg_file = write_txt.split('/')[0]
    if not os.path.exists(cfg_file):
        os.mkdir(cfg_file)
    
    if os.path.exists(write_txt):
        file=open(write_txt, 'w')

    run_convert(all_classes, train_img, train_annotation, yolo_path, write_txt)