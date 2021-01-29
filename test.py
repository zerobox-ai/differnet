'''This is the repo which contains the original code to the WACV 2021 paper
"Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows"
by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.
For further information contact Marco Rudolph (rudolph@tnt.uni-hannover.de)'''

import config as c
from train import *
from utils import load_datasets, make_dataloaders
import time
import gc
import json

_, _, test_set = load_datasets(c.dataset_path, c.class_name, test=True)
_, _, test_loader = make_dataloaders(None, None, test_set, test=True)

save_name_pre = '{}_{}_{:.2f}_{:.2f}_{:.2f}_{:.2f}'.format(c.modelname, c.rotation_degree,
                                               c.crop_top, c.crop_left, c.crop_bottom, c.crop_right)

model = torch.load("models/" + save_name_pre + ".pth", map_location=torch.device('cpu'))

with open('models/' + save_name_pre + '.json') as jsonfile:
    model_parameters = json.load(jsonfile)

time_start = time.time()
test(model, model_parameters, test_loader)
time_end = time.time()
time_c = time_end - time_start  # 运行所花时间

print("test time cost: {:f} s".format(time_c))