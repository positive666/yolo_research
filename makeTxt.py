"""

Usage:
    $ python makeTxt.py --path 'DataPath'

"""
import os
import random
import argparse




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/all_labelData', help='data path')
    opt = parser.parse_args()
    print(opt)
    return opt
    
    
trainval_percent = 0.9
train_percent = 0.9
opt = parse_opt()
xmlfilepath = opt.path + '/Annotations'
txtsavepath = opt.path + '/ImageSets'
total_xml = os.listdir(xmlfilepath)


num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open(opt.path + '/ImageSets/trainval.txt', 'w')
ftest = open(opt.path + '/ImageSets/test.txt', 'w')
ftrain = open(opt.path + '/ImageSets/train.txt', 'w')
fval = open(opt.path + '/ImageSets/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()


