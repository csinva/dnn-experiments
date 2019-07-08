# understanding how deep learning works
this repo contains code for running a variety of different experiments attempting to understand deep learning via empirical experiments

# organization
- each folder contains a readme with code documentation, as well as comments in the code
- the vision_fit and vision_analyze folders detail a number of experiments on multilayer perceptrons and convolutional neural networks using various datasets including MNIST, CIFAR, and custom datasets
- the sparse_coding folder contains code for running and analyzing sparse coding on different sets of images
- the mog folder contain code examples for fitting synthetic datasets generated as mixtures of Gaussians
- the poly_fit folder contains code for fitting simple 1D polynomials
- the scripts folder contains scripts for launching jobs on a slurm cluster
- the eda folder contains minimum working examples for simple setups with various pytorch and scikit-learn functions

# requirements
- the code is all tested in python3 and pytorch 1.0

# running
- the `scripts` folder contains sample slurm scripts for launching jobs on a cluster
- most of the experiments are time-consuming and should be parallelized over many machines
- to do so, ssh into one of the scf nodes (e.g. legolas) and run ```module load python```
- set the parameters you want to sweep as lists in one of the submit*.py files
- then run this file and it will automatically launch slurm jobs for each set of parameters
