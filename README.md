# Highly Scalable Deep Convolutional Neural Networks with Sparsely Aggregated Internal Connections

**E6876 Sparse and Low-dimensional Models for High Dimensional Data - Final Project (Spring 2022)**

**Group Members:**

Siddharth Nijhawan (sn2951)

Sushant Tiwari (st3425)

Aman Anand (aa4821)

**Summary:** The concept of sparsity in neural network parameters and architectures has been revolutionizing the world of deep convolution neural networks. Building a neural network architecture through sparse aggregation via internal skip connections can help not only in the significant reduction in training time but also in the computational complexity and convergence. Present state- of-the-art network architectures like ResNet and DenseNet focus on skip connections but not in an efficient manner. This leaves lot of scope in implementing sparsity with a focus on the selected layer-wise inputs for sparse aggregation in these architectures. Our work builds on this sparse aggregation concept at the architecture level by choosing only selected inputs at a given depth for propagating the most important information to the successive layers. Through our computational experiments and in-depth analysis, we propose an architecture design which combines the residual skip connections with the selected parameter sparse aggregation to train highly deep convolution neural networks. Through our results on image classification datasets CIFAR-10 and CIFAR-100, we argue that our design implementation gives better performance in terms of training time, training parameters and overall classification accuracy for a given layer depth and the feature growth rate. Also, we analyze the performance of our implementation of the architecture design called SparseNet with DenseNets on CIFAR-10 and CIFAR-100 datasets through experimentation and with ResNet and FractalNet analytically. We compare these architectures in terms of computational complexity by performing in-depth sparse aggregation analysis. We build a highly robust and scalable model by only aggregating a selected/sparse penultimate outputs sets for a current depth and thereby achieving reduction in number of incoming input links. This decrease in the link size from linear to logarithmic helps in a significant improvement in the above mentioned performance metrics.

**Instructions:** 

Execute the cells of notebooks densenet_pytorch_cifar10.ipynb and sparsenet_pytorch_cifar10.ipynb sequentially to train the models and generate results of proposed work on CIFAR10 dataset. Similarly, densenet_pytorch_cifar100.ipynb and sparsenet_pytorch_cifar100.ipynb notebooks generate results for CIFAR-100 dataset. Datset is automatically downloaded during the execution and placed in data/ folder.

**Code structure:**

utils.py - contains class and function definitions for various types of layers used in creating the neural network (single_layer, bottleneck_layer, flatten_layer, dense_stage, etc.)

model_architecture.py - contains the master class for defining the neural network model for our proposed work (densenet as well as sparsenet)

**Python Packages Installation:**

```
pip install numpy torch torchvision
```

**Pre-trained models can be accessed inside models/ folder here:** https://drive.google.com/file/d/1yzp_TR0xy5ADVO-eQm8OPbA1IP7VSWYl/view?usp=sharing
