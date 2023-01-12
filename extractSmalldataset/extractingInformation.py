import torch
import sys
import os
import warnings
import numpy as np
import torch.utils.data
import random
from datetime import datetime
import json
import random
import shutil
import pathlib
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path
#def to_prefix(source):
#    return ('_').join(source.split('_')[:-1])+'_' ##removing post-tag of dynamic
##20090316_1008_366_clc_mode.npy
##20090316_1008_366_clc_vec.npy
##20090316_1008_366_dynamic.npy
##20090316_1008_366_static.npy

def copyfile(source, dest):
    prefix = source.split('/')
    dest = dest + '/'+('/').join(prefix[-4:-1])+'/'
    command = 'cp '+source + ' ' +dest
    os.system(command)
    commandcpy = command.replace('dynamic', 'static')
    os.system(commandcpy)
    commandcpy = command.replace('dynamic', 'clc_mode')
    os.system(commandcpy)
    commandcpy = command.replace('dynamic', 'clc_vec')
    os.system(commandcpy)

#    print(source)
#    print(dest)
#    print(name)
#    print(source)
#    print(dest)
#    exit(0)
#    p = pathlib.Path(item[0])
#    source = pathlib.Path(*p.parts[:-1])
#    name = p.stem.split('_')[:-1]
#    prefix = '_'.join(name)
#    prefixPath = pathlib.Path(source).glob(prefix+'*')
#    for filename in prefixPath:
#        pto = pathlib.Path(filename)
#        to = dest / pathlib.Path(*pto.parts[4:])
#        From = str(filename)
#        To = str(to)
#        command = 'cp '+From+' '+To
#        shutil.copyfile(filename, dest / to)
#        os.system(command)
seed = int(sys.argv[1])
print(seed, "<--- seed")
source = '/home/joel.chacon/tmp/datasets_grl/'
source = '/home/joel.chacon/tmp/selectingSampling/data/datasets_grl1/datasets_grl/'
dest = 'datasets_grl/'
random.seed(seed)
dataset_root = Path(source)
min_max_file = dataset_root / 'minmax_clc.json'
variable_file = dataset_root / 'variable_dict.json'
access_mode = 'spatiotemporal'
dataset_path = dataset_root / 'npy' / access_mode
neg_pos_ratio = 2

positives_list = list((dataset_path / 'positives').glob('*dynamic.npy'))
positives_list = list(zip(positives_list, [1] * (len(positives_list))))
val_year = 2019
test_year1 = 2020 
test_year2 = 2021 

train_positive_list = [x for (x, y) in positives_list if int(x.stem[:4]) < val_year]
val_positive_list = [x for (x, y) in positives_list if int(x.stem[:4]) == val_year]
test1_positive_list = [x for (x, y) in positives_list if int(x.stem[:4]) == test_year1]
test2_positive_list = [x for (x, y) in positives_list if int(x.stem[:4]) == test_year2]

negatives_list = list((dataset_path / 'negatives_clc').glob('*dynamic.npy'))
negatives_list = list(zip(negatives_list, [0] * (len(negatives_list))))

train_negative_list = random.sample([ str(x) for (x, y) in negatives_list if int(x.stem[:4]) < val_year], len(train_positive_list) * neg_pos_ratio)

val_negative_list = random.sample([str(x) for (x, y) in negatives_list if int(x.stem[:4]) == val_year], len(val_positive_list) * neg_pos_ratio)

test1_negative_list = random.sample([str(x) for (x, y) in negatives_list if int(x.stem[:4]) == test_year1], len(test1_positive_list) * neg_pos_ratio)

test2_negative_list = random.sample([str(x) for (x, y) in negatives_list if int(x.stem[:4]) == test_year2], len(test2_positive_list) * neg_pos_ratio)


#####getting a small portion of it....

train_positive_list = random.sample(train_positive_list, int(len(train_positive_list)*0.1))
val_positive_list = random.sample(val_positive_list, int(len(val_positive_list)*0.1))
test1_positive_list = random.sample(test1_positive_list, int(len(test1_positive_list)*0.1))
test2_positive_list = random.sample(test2_positive_list, int(len(test2_positive_list)*0.1))

train_negative_list = random.sample(train_negative_list, int(len(train_negative_list)*0.1))
val_negative_list = random.sample(val_negative_list, int(len(val_negative_list)*0.1))
test1_negative_list = random.sample(test1_negative_list, int(len(test1_negative_list)*0.1))
test2_negative_list = random.sample(test2_negative_list, int(len(test2_negative_list)*0.1))



cont = 0
for item in train_negative_list:
   copyfile(item, dest)
   cont +=1
print("train negative: "+str(cont))
cont = 0
for item in val_negative_list:
   copyfile(item, dest)
   cont +=1
print("val negative: "+str(cont))
cont = 0
for item in test1_negative_list:
   copyfile(item, dest)
   cont +=1
print("test1 negative: "+str(cont))
cont = 0
for item in test2_negative_list:
   copyfile(item, dest)
   cont +=1
print("test2 negative: "+str(cont))


for item in train_positive_list:
   copyfile(item, dest)
   cont +=1
print("train positive: "+str(cont))
cont = 0
for item in val_positive_list:
   copyfile(item, dest)
   cont +=1
print("val positive: "+str(cont))
cont = 0
for item in test1_positive_list:
   copyfile(item, dest)
   cont +=1
print("test1 positive: "+str(cont))
cont = 0
for item in test2_positive_list:
   copyfile(item, dest)
   cont +=1
print("test2 positive: "+str(cont))
