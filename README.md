# ModelForOAMLight
This Repo contributes to measure OAM spectrum of structured light via hybrid optoelectronic neural networks. Detecting OAM spectrum is a crucial ingredient for almost all aspects of its related applications. This proposed method settles this pivotal task in an accurate, fast, robust, concise and single-shot manner. It can be an optimal solution to OAM measurement problem compared with previous endeavors, openning substantial OAM-based fundamental discoveries as well as applications such as high-speed communications.

To dig out more details, please refer to http://arxiv.org/abs/2205.15045.

If you find this repo helpful for your research, please also cite this paper (http://arxiv.org/abs/2205.15045).

To start with, please download this package and install PyTorch (recommend PyTorch 1.10.0 with Python 3.7 and CUDA 11.5.0 environment).

HybridNetwork.py is the constructed model.

train.py is the codes for optimizing the hybrid network.

Visualization folder contains codes about how we interpret the optoelectronic model. (Model and dataset are uploaded to google drive.

infer.py is the codes for blindly testing the trained model. We upload six pretrained model in file folder 'final_model', which one can implement and check the test results. Several test sets are put into file folder 'Dataset', including experimental sets and simulation sets. One can load all of them only by changing the root within infer.py to validate the generalization ability of the final trained models. The pretrained models and test sets can be accessed from here: https://drive.google.com/drive/folders/17p77ykOqnBdEXlCnbI5GEfhsjBSR3Col?usp=sharing, https://drive.google.com/drive/folders/1jAjWbkqc3H-j_FNPIieCfo5R3RM-S8s6?usp=sharing

Contact: Hao Wang, h-wang20@mails.tsinghua.edu.cn
