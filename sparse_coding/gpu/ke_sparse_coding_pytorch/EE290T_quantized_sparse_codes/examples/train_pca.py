"""
Train an PCA dictionary
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

from training.pca import train_dictionary as pca_train
from analysis_transforms import invertible_linear
from utils.image_processing import create_patch_training_set
from utils.plotting import TrainingLivePlot
from utils.plotting import display_dictionary

RUN_IDENTIFIER = 'test_PCA'

NUM_IMAGES_TRAIN = 1000000
PATCH_HEIGHT = 16
PATCH_WIDTH = 16

CODE_SIZE = PATCH_HEIGHT * PATCH_WIDTH

# Arguments for dataset and logging
parser = argparse.ArgumentParser()
parser.add_argument("data_id",
    help="Name of the dataset (currently allowable: " +
         "Field_NW_whitened, Field_NW_unwhitened)")
parser.add_argument("data_filepath", help="The full path to dataset on disk")
script_args = parser.parse_args()

torch_device = torch.device('cuda:1')
torch.cuda.set_device(1)
# otherwise can put on 'cuda:0' or 'cpu'

# manually create large training set with one million whitened patches
one_mil_image_patches = create_patch_training_set(
    ['patch', 'center'], (PATCH_HEIGHT, PATCH_WIDTH),
    NUM_IMAGES_TRAIN, 1, edge_buffer=5, dataset=script_args.data_id,
    datasetparams={'filepath': script_args.data_filepath,
                   'exclude': []})['batched_patches']

#################################################################
# save these to disk if you want always train on the same patches
# or if you want to speed things up in the future
#################################################################
# pickle.dump(one_mil_image_patches, open('/media/expansion1/spencerkent/Datasets/Field_natural_images/one_million_patches_whitened_June25.p', 'wb'))

# one_mil_image_patches = pickle.load(open(
#     '/media/expansion1/spencerkent/Datasets/Field_natural_images/one_million_patches_whitened_June25.p', 'rb')).astype('float32')

# we are going to 'unbatch' them because pca will train on the whole dataset
# at once
image_patches_gpu = torch.from_numpy(
    one_mil_image_patches.transpose((0, 2, 1)).reshape(
      (-1, PATCH_HEIGHT*PATCH_WIDTH)).T).to(torch_device)

pca_dictionary = pca_train(image_patches_gpu)
codes = invertible_linear.run(image_patches_gpu, pca_dictionary, orthonormal=True)

plots = display_dictionary(pca_dictionary.cpu().numpy(),
                           16, 16, 'PCA-determined basis functions')
plt.show()
