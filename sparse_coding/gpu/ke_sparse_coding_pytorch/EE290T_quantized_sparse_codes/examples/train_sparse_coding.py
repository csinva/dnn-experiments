"""
Train a sparse coding dictionary. These settings for Field natural images dset
"""
import sys
import os
examples_fullpath = os.path.dirname(os.path.abspath(__file__))
toplevel_dir_fullpath = examples_fullpath[:examples_fullpath.rfind('/')+1]
sys.path.insert(0, toplevel_dir_fullpath)

import argparse
import pickle
import numpy as np
from matplotlib import pyplot as plt
import torch

from training.sparse_coding import train_dictionary as sc_train
from utils.plotting import TrainingLivePlot
from utils.image_processing import create_patch_training_set

RUN_IDENTIFIER = 'test_sparse_coding'

BATCH_SIZE = 250
NUM_BATCHES = 4000  # 1 million patches total
PATCH_HEIGHT = 16
PATCH_WIDTH = 16

CODE_SIZE = 256
NUM_EPOCHS = 30

SC_PARAMS = {
    'num_epochs': NUM_EPOCHS,
    'code_inference_algorithm': 'fista',
    'inference_param_schedule': {
      0: {'sparsity_weight': 0.1, 'max_num_iters': 50},
      10*NUM_BATCHES: {'sparsity_weight': 0.1, 'max_num_iters': 100},
      20*NUM_BATCHES: {'sparsity_weight': 0.1, 'max_num_iters': 200}},
    'dictionary_update_algorithm': 'sc_cheap_quadratic_descent',
    'dict_update_param_schedule': {
      0: {'stepsize': 0.05, 'num_iters': 1},
      10*NUM_BATCHES: {'stepsize': 0.01, 'num_iters': 1},
      20*NUM_BATCHES: {'stepsize': 0.005, 'num_iters': 1}},
    'training_visualization_schedule': {0: None, 1000: None, 2000: None}}
SC_PARAMS['training_visualization_schedule'].update(
    {NUM_BATCHES*x: None for x in range(NUM_EPOCHS)})

# Arguments for dataset and logging
parser = argparse.ArgumentParser()
parser.add_argument("data_id",
    help="Name of the dataset (currently allowable: " +
         "Field_NW_whitened, Field_NW_unwhitened)")
parser.add_argument("data_filepath", help="The full path to dataset on disk")
parser.add_argument("-l", "--logfile_dir",
                    help="Optionally checkpoint the model here")
script_args = parser.parse_args()

if script_args.logfile_dir is not None:
  SC_PARAMS['checkpoint_schedule'] = {'checkpoint_folder_fullpath':
      script_args.logfile_dir + RUN_IDENTIFIER,
      NUM_BATCHES: None, 10*NUM_BATCHES: None, 20*NUM_BATCHES:None}

torch_device = torch.device('cuda:1')
torch.cuda.set_device(1)
# otherwise can put on 'cuda:0' or 'cpu'

# manually create large training set with one million whitened patches
one_mil_image_patches = create_patch_training_set(
    ['patch'], (PATCH_HEIGHT, PATCH_WIDTH), BATCH_SIZE, NUM_BATCHES,
    edge_buffer=5, dataset=script_args.data_id,
    datasetparams={'filepath': script_args.data_filepath,
                   'exclude': []})['batched_patches']

#################################################################
# save these to disk if you want always train on the same patches
# or if you want to speed things up in the future
#################################################################
# pickle.dump(one_mil_image_patches, open('/media/expansion1/spencerkent/Datasets/Field_natural_images/one_million_patches_October24.p', 'wb'))

# one_mil_image_patches = pickle.load(open(
#     '/media/expansion1/spencerkent/Datasets/Field_natural_images/one_million_patches_October24.p', 'rb')).astype('float32')

# send ALL image patches to the GPU
image_patches_gpu = torch.from_numpy(one_mil_image_patches).to(torch_device)

# create the dictionary Tensor on the GPU
sparse_coding_dictionary = torch.randn((PATCH_HEIGHT*PATCH_WIDTH, CODE_SIZE),
                                       device=torch_device)
sparse_coding_dictionary.div_(sparse_coding_dictionary.norm(p=2, dim=0))

# Create the LivePlot object
liveplot_obj = TrainingLivePlot(
    dict_plot_params={'total_num': CODE_SIZE, 'img_height': PATCH_HEIGHT,
                      'img_width': PATCH_WIDTH, 'plot_width': 16,
                      'plot_height': 16, 'renorm imgs': True,
                      'display_ordered': True},
    code_plot_params={'size': CODE_SIZE})

SC_PARAMS['training_visualization_schedule']['liveplot_object_reference'] = liveplot_obj

print("Here we go!")
sc_train(image_patches_gpu, sparse_coding_dictionary, SC_PARAMS)
