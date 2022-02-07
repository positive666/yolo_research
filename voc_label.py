"""

Usage:
    $ python voc_label.py --path 'DataPath'

"""

# encoding=utf-8
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import argparse


sets = ['train', 'test', 'val']
classes = ['class_1','class2']  #define your classes

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/all_labelData', help='data path')
    opt = parser.parse_args()
    print(opt)
    return opt
 
    
def convert(size, box): 
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)

opt = parse_opt()


def convert_annotation(image_id):

    in_file = open(opt.path + '/Annotations/%s.xml' % (image_id), encoding='utf-8')
    out_file = open(opt.path + '/labels/%s.txt' % (image_id), 'w', encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    
    if size != None:
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1 :
                continue

            cls_id = classes.index(cls)

            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))

            print(image_id, cls, b, i)
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
print(wd)


for image_set in sets:
    if not os.path.exists(opt.path + '/labels/'):
        os.makedirs(opt.path + '/labels/')
    image_ids = open(opt.path + '/ImageSets/%s.txt' % (image_set)).read().strip().split()
    list_file = open(opt.path + '/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write(opt.path + '/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
