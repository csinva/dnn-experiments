EE290T Fall 2018 Project: Toward Efficient Quantization of Sparse Codes
=======================================================================
This is a repository that contains some implementations relevant to our project for EE290T.

It currently just contains an implementation of canonical Sparse Coding, ICA, and PCA along with a few utilities for processing image data.
The implementation is in the PyTorch GPGPU framework and should be relatively performant in terms of wall-clock time for training
and inference. The modular organization is to stress the interchangibility of different techniques for code inference (analysis)
and reconstruction (synthesis).

## Installation Instructions
1. Clone the repository
2. Download the data you want to run these models on (for illustration try
   David Field's Images of the Northwest - Spencer can provide).
3. Make a logfile directory somewhere if you want to save the state of the model
   during training
4. Look at one of the scripts in the examples/ directory. Invoke one of these with
   commandline arguments that specify the dataset and optionally the logfile directory
   (try `python train_sparse_coding -h` for more details)

## Dependencies
* NumPy
* Matplotlib
* SciPy (stats toolbox), just for plotting (can be ommitted)
* json
* torch (0.4+, not tested current release (1.0) though)


