import math
# === parameters for main()
# global parameters
batch_size = 200
lr = 5.0e-4
num_classes = 21
num_layers = 5
wd = 0.8e-2
step_size = 15
gamma = 0.8
scale = 11
extension = 2
T = 0.1
# === parameters for hybrid model
size = 200
distance = 40
distance_pro = 40
distance_det = 60
ls = 106.7
fs = 1.0 / (2 * ls)
wl = 1
k = 2.0 * math.pi / wl
paddings = 100
kernel_size = 4
stride = 4
hidden = 2500
dim_in = 2500
dim_out = 21
range = dim_out
# ==== visualization model
win_size = 1
map_size = int((size + kernel_size) / stride - 1)
shift = 3
thershold = 0.015
width = depth = 0.3


