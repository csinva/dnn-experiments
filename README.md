# Understanding how deep learning works
- This repo contains code for running a variety of different relatively simple experiments attempting to understand deep learning.

# Organization
- The vision_fit and vision_analyze folders detail a number of experiments on multilayer perceptrons and convolutional neural networks using various datasets including MNIST, CIFAR, and custom datasets
- The sparse_coding folder contains code for running sparse coding on different sets of images
- The mog_fit and mog_analyze folders contain code examples for fitting synthetic datasets generated as mixtures of Gaussians
- The poly_fit folder contains code for fitting simple 1D polynomials
- The scripts folder contains scripts for launching jobs on a slurm cluster
- The eda folder contains minimum working examples for simple setups with various pytorch and scikit-learn functions

# Requirements
- The code is all run in python3 and pytorch 1.0

# Running
- Most of the experiments are extremely time consuming and should be parallelized over many machines
- To do so, ssh into one of the scf nodes (e.g. legolas) and run ```module load python```
- set the parameters you want to sweep as lists in one of the submit*.py files
- then run this file and it will automatically launch slurm jobs for each set of parameters