# Author Hao Wang & Ziyu Zhan
# OAM spectrum detection based on hybrid optoeletronic neural network
#  ============================
import torch
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import os
import time
import batch_loader as bl
import HybridNetwork
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./ResultRepo')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# global parameters
batch_size = 150
batch_size1 = 200
lr = 1.0e-3
num_classes = 21
num_layers = 5
wd = 0.8e-2
step_size = 15
gamma = 0.8
scale = 11
extension = 2
T = 0.1
model_name = 'layers=%d_classes=%d_lr=%.3f_T=%.3f_sc=%d_ex=%d' % (num_layers, num_classes, lr, T, scale, extension)
print('model:{}'.format(model_name))
model_dir = "Models/" + model_name
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
#------------path for saved models : save 3 models
model_val1 = "Models/" + model_name + "/best_model1.pth"
path_best1 = os.path.abspath(model_val1)
model_val2 = "Models/" + model_name + "/best_model2.pth"
path_best2 = os.path.abspath(model_val2)
model_val3 = "Models/" + model_name + "/best_model3.pth"
path_best3 = os.path.abspath(model_val3)
model_train1 = "Models/" + model_name + "/best_train1.pth"
path_train1 = os.path.abspath(model_train1)
model_train2 = "Models/" + model_name + "/best_train2.pth"
path_train2 = os.path.abspath(model_train2)
model_train3 = "Models/" + model_name + "/best_train3.pth"
path_train3 = os.path.abspath(model_train3)
path_best = [path_best1, path_best2, path_best3]
path_train = [path_train1, path_train2, path_train3]
model_temp = "Models/" + model_name + "/temp_model1.pth"
path_temp = os.path.abspath(model_temp)
# load data
root1 = r'C:\DataSim\train'
root2 = r'C:\DataSim\val'
transform = transforms.Compose([transforms.ToTensor, transforms.RandomRotation((0, 360))])
train_dataset = bl.TrainDataset(root1, transform=transform)
data_train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = bl.ValDataset(root2, transform=transform)
data_val = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
# define the model, loss function and optimizer
model = HybridNetwork.Net(num_layers=num_layers)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(device)
    model.cuda()
else:
    print('no gpu available')
criterion = torch.nn.SmoothL1Loss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
min_loss, min_train = float('inf'), float('inf')
idx_best, idx_train = 0, 0
# train & val
num_epochs = 300
check = 16
iters = 0
for epoch in range(num_epochs):
    start0 = time.time()
    running_loss = 0.0
    for i, (train_input, train_labels) in enumerate(data_train, 1):
        train_input = train_input.cuda()  # b h w
        train_input = torch.squeeze(train_input)
        train_labels = train_labels.cuda()  # b 81 1
        train_labels = torch.squeeze(train_labels).float()
        optimizer.zero_grad()
        eff_train = model(train_input)
        loss = criterion(eff_train, train_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % check == 0:
            total = 0
            val_loss = 0.0
            model.eval()
            for j, (val_input, val_labels) in enumerate(data_val, 1):
                val_input = val_input.cuda()
                val_input = torch.squeeze(val_input)
                val_input = val_input
                val_labels = val_labels.cuda()
                val_labels = torch.squeeze(val_labels)
                with torch.no_grad():
                    eff_val = model(val_input)
                    loss2 = criterion(eff_val, val_labels)
                val_loss += loss2.item()
            validation_loss = val_loss / len(data_val)
            training_loss = running_loss / check
            print('[{}, {}] train_loss = {:.5f} val_loss = {:.5f}'.format(epoch + 1, i, training_loss, validation_loss))
            writer.add_scalar('train_loss', training_loss, iters)
            writer.add_scalar('val_loss', validation_loss, iters)

            if validation_loss < min_loss:
                print('saving a lowest loss model: best_model')
                min_loss = validation_loss
                torch.save({
                    'Model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'min_loss': min_loss,
                    'min_train': min_train,
                    'idx_best': idx_best
                }, path_best[idx_best])
                idx_best = idx_best + 1
                idx_best = idx_best % 3
            if training_loss < min_train:
                print('saving a lowest train_loss model: best_train')
                min_train = training_loss
                torch.save({
                    'Model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'min_loss': min_loss,
                    'min_train': min_train,
                    'idx_train': idx_train
                }, path_train[idx_train])
                idx_train = idx_train + 1
                idx_train = idx_train % 3
            iters += 1
            running_loss = 0.0
    print('one epoch time %.2f sec' % (time.time() - start0))