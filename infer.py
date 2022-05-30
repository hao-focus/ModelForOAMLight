# Author Hao Wang & Ziyu Zhan
# OAM spectrum detection based on hybrid optoeletronic neural network
#  ============================
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import batch_loader2 as bl
import HybridNetwork
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# global parameters
batch_size = 2
lr = 5.0e-4
num_classes = 21
num_layers = 5
wd = 0.8e-2
step_size = 15
gamma = 0.8
scale = 11
extension = 2
T = 0.1
#------------path for saved models : save 3 models
model_name = 'final_model'
model_val1 = model_name + "/best_model1.pth"
path_best1 = os.path.abspath(model_val1)
model_val2 = model_name + "/best_model2.pth"
path_best2 = os.path.abspath(model_val2)
model_val3 = model_name + "/best_model3.pth"
path_best3 = os.path.abspath(model_val3)
model_train1 = model_name + "/best_train1.pth"
path_train1 = os.path.abspath(model_train1)
model_train2 = model_name + "/best_train2.pth"
path_train2 = os.path.abspath(model_train2)
model_train3 = model_name + "/best_train3.pth"
path_train3 = os.path.abspath(model_train3)
path_best = [path_best1, path_best2, path_best3]
path_train = [path_train1, path_train2, path_train3]
def bar_show(img, x):
    img_np = img.numpy()
    plt.bar(x, img_np, color='green')
    plt.show()
# load the data
root = r'E:\dl_project\D2NN_OAM_Spectrum\GithubRepo\Dataset\ExpDataset\test_MulMode'
# root = r'E:\dl_project\D2NN_OAM_Spectrum\GithubRepo\Dataset\ExpDataset\test_MulMode'
# root = r'E:\dl_project\D2NN_OAM_Spectrum\GithubRepo\Dataset\ExpDataset\test_MulMode'
# root = r'E:\dl_project\D2NN_OAM_Spectrum\GithubRepo\Dataset\ExpDataset\test_MulMode'
# root = r'E:\dl_project\D2NN_OAM_Spectrum\GithubRepo\Dataset\ExpDataset\test_MulMode'
# root = r'E:\dl_project\D2NN_OAM_Spectrum\GithubRepo\Dataset\ExpDataset\test_MulMode'
transform = transforms.Compose([transforms.ToTensor])
test_dataset = bl.TrainDataset(root, transform=transform)
data_test = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
model = HybridNetwork.Net(num_layers=num_layers)
criterion = torch.nn.SmoothL1Loss(reduction='sum')
checkpoint = torch.load(path_best[0], map_location='cpu')  # or path_best
model.load_state_dict(checkpoint['Model_state_dict'])
model.cuda()
model.eval()
c1 = torch.arange(-10, 11, 1)
for i, (test_input, test_labels) in enumerate(data_test, 1):
    test_input = test_input.cuda()
    test_input = torch.squeeze(test_input)
    test_labels = test_labels.cuda()
    test_labels = torch.squeeze(test_labels)
    with torch.no_grad():
        ccd_signal = model(test_input)
    bar_show(test_labels[1, :].cpu().detach(), c1)
    bar_show(ccd_signal[1, :].cpu().detach(), c1)