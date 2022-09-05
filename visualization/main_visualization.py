# Author Ziyu Zhan
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import batch_loader as bl
import visualization_HybridNetwork
import scipy.io as sio
import visual_paras as para
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
model_name = 'final_model_visualization.pth'
model_val = "final_Model/" + model_name
path_best = os.path.abspath(model_val)
# plot initialization
x = torch.arange(0, para.map_size, 1)
y = torch.arange(0, para.map_size, 1)
xx, yy = np.meshgrid(x, y)
X, Y = xx.ravel(), yy.ravel()

def bar_show(img, x):
    img_np = img.numpy()
    plt.bar(x + 0.35, img_np, width=0.6, color='peru')
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Feature distribution', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.GnBu, vmin=0, vmax=25)
    plt.colorbar()
    plt.show()
# load the data
root = r'' #  the absolute root of test set 'Dataset_ForVisualization'
transform = transforms.Compose([transforms.ToTensor])
test_dataset = bl.TrainDataset(root, transform=transform)
data_test = DataLoader(dataset=test_dataset, batch_size=para.batch_size, shuffle=False)
model_numerator = visualization_HybridNetwork.Net(num_layers=para.num_layers)
checkpoint = torch.load(path_best, map_location='cpu')
model_numerator.load_state_dict(checkpoint['Model_state_dict'])
model_numerator.cuda()
model_numerator.eval()

for i, (test_input, test_labels) in enumerate(data_test, 1):
    test_input = test_input
    test_input = test_input.cuda()
    test_input = torch.squeeze(test_input)
    with torch.no_grad():
        result,  ch_map, histo = model_numerator(test_input, para.batch_size)

        histogram = torch.arange(para.range)
        x_range = torch.arange(para.range)
        histogram = histo[1:]
        bar_show(histogram, x_range)

        ch_map = ch_map.cpu().numpy()
        name = 'map_1times1_updated.mat'
        sio.savemat(name, {'map': ch_map})
        # plot the characteristic graph
        result = torch.flatten(result, start_dim=0, end_dim=-1).float()
        result = result.numpy()
        classes = np.arange(0, para.map_size, 1, dtype=np.int16)
        result = result.reshape((para.map_size, para.map_size))
        plot_confusion_matrix(result, classes, normalize=False, title='Characteristic Graph')





