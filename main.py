'''This is the repo which contains the original code to the WACV 2021 paper
"Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows"
by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.
For further information contact Marco Rudolph (rudolph@tnt.uni-hannover.de)'''

import config as c
from train import *
from utils import load_datasets, make_dataloaders
import time
import gc

#train_set, validate_set, _ = load_datasets(c.dataset_path, c.class_name)
# train_loader, validate_loader, _ = make_dataloaders(train_set, validate_set, None)
train_set, validate_set, _ = load_datasets(c.dataset_path, c.class_name, test=True)
train_loader, validate_loader, test_set_ = make_dataloaders(train_set, validate_set,testset=test_set, test=True)


time_start = time.time()
model, model_parameters = train(train_loader, validate_loader)
#model, model_config = train(train_loader, None)
time_end = time.time()
time_c = time_end - time_start  # 运行所花时间
print("train time cost: {:f} s".format(time_c))

# free memory
del train_set
del validate_set
del train_loader
del validate_loader

gc.collect()
torch.cuda.empty_cache()