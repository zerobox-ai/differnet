import pickle
import numpy as np
import config as c
from train import *
from utils import load_datasets, make_dataloaders
import time
import gc
import json

_, _, predict_set = load_datasets(c.dataset_path, 'predict', test=True)
_, _, predict_loader = make_dataloaders(None, None, predict_set, test=True)

model = torch.load("models/" + c.modelname + "", map_location=torch.device('cpu'))

with open('models/' + c.modelname + '.json') as jsonfile:
    model_parameters = json.load(jsonfile)


model.to(c.device)
model.eval()

test_z = list()
test_labels = list()

with torch.no_grad():
    for i, data in enumerate(predict_loader):
        inputs, labels = preprocess_batch(data)
        frame = int(predict_loader.dataset.imgs[i][0].split('frame', 1)[1].split('-')[0])
        print(f"i={i}: frame#={frame}, labels={labels.cpu().numpy()[0]}, size of inputs={inputs.size()}")
        z = model(inputs)
        x = model.decoder(z)
        test_z.append(z)
        test_labels.append(t2np(labels))

alpha = 0

# Split labelled dataset based on attribute, say bottle without background
#X_positive, X_negative = split(X_labelled)

# Obtain average encodings of positive and negative inputs
#z_positive = np.mean([model.encode(x) for x in X_positive])
#z_negative = np.mean([model.encode(x) for x in X_negative])

# Get manipulation vector by taking difference
#z_manipulate = z_positive - z_negative

# Manipulate new x_input along z_manipulate, by a scalar alpha \in [-1,1]
#z_input = model.encode(x_input)
#x_manipulated = model.decode(z_input + alpha * z_manipulate)