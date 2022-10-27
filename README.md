# **Explainable Federated Learning Method for Privacy-preserving Long-tailed Visual Recognition**

This repository contains code for the paper:
**Explainable Federated Learning Method for Privacy-preserving Long-tailed Visual Recognition**  
> Yu-Chi Wang, Jen-Sheng Tsai, and Yau-Hwang Kuo  
> Institute of Medical Informatics, National Cheng Kung University  
>


**Abstract:** The visual recognition application of the model will encounter several bottlenecks. First, real-world data face privacy issues. Second, the data usually presents a long-tailed distribution. Finally, there is the black box problem. We propose an explainable federated learning method that combines interpretable models, two-stage learning, and a novel model aggregation strategy for privacy-preserving long-tailed visual recognition. Specifically, we use ProtoTree as the basic architecture because it can maintain good performance while being interpretable through combining deep learning and soft decision tree. Then, we separate model training into client training and server training. The representation learning is trained by the client. An intuitive and easy-to-understand tree-based classifier is re-trained by the server with a resampling strategy. Furthermore, a novel federated aggregation strategy aggregates models by referring to client-side data distributions to improve training performance. Experimental results demonstrate that our architecture can overcome the problem of long-tailed distribution and obtain an intuitive, understandable, and interpretable soft decision tree without exposing private data while achieving the highest accuracy rate. Moreover, the performance of our method surpasses those of existing federated long-tailed learning methods for long-tailed classification.



### Dependencies

- python 3.6.12 (Anaconda)
- PyTorch 1.7.0+cu110
- torchvision 0.8.1+cu110
- CUDA 11.4
- cuDNN 11.1.105



### Dataset

- CIFAR10-LT
- CIFAR100-LT
- Place-LT
    - Download the [Places_365](http://places2.csail.mit.edu/download.html).

### Reference
* prototree：https://github.com/M-Nauta/ProtoTree
* place-LT(split train and test file)：https://github.com/facebookresearch/classifier-balancing

