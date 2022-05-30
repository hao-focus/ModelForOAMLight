# ModelForOAMLight
This Repo contributes to measure OAM spectrum of structured light via hybrid optoelectronic neural networks. Detecting OAM spectrum is a crucial ingredient for almost all aspects of its related applications. This proposed method settles this pivotal task in an accurate, fast, robust, concise and single-shot manner. It can be an optimal solution to OAM measurement problem compared with previous endeavors, openning substantial OAM-based fundamental discoveries as well as applications such as high-speed communications.

To dig out more details, refer to xxx.

If this repo is helpful, please also cite xxx.

HybridNetwork.py is the constructed model.

train.py is the codes for optimizing the hybrid network.

infer.py is the codes for blindly testing the trained model. We upload six trained model in file folder 'final_model', which one can implement and check the test results. Several test sets are put into file folder 'Dataset', including experimental sets and simulation sets. One can load all of them only by changing the root within infer.py to validate the generalization ability of the final trained models.
