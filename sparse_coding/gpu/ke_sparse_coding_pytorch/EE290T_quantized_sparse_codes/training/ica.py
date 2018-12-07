"""
This implements the ICA dictionary learning algorithm
"""

import time
import os
import pickle
import json
from matplotlib import pyplot as plt
import torch

def train_dictionary(image_dataset, init_dictionary, all_params):
  """
  Train an ICA dictionary

  Parameters
  ----------
  image_dataset : torch.Tensor OR torch.Dataloader
      We make __getitem__ calls to either of these iterables and have them
      return us a batch of images. If image_dataset is a torch Tensor, that
      means ALL of the data is stored in the CPU's RAM or in the GPU's RAM. The
      choice of which will have already been made when the Tensor is created.
      The tensor is an array of size (k, n, b) where k is the total number of
      batches, n is the (flattened) size of each image, and b is the size of
      an individual batch. If image_dataset is a torch.DataLoader that means
      each time we make a __getitem__ call it will return a batch of images that
      it has fetched and preprocessed from disk. This is done in cpu
      multiprocesses that run asynchronously from the GPU. If the whole dataset
      is too large to be loaded into memory, this is really our only option.
  init_dictionary : torch.Tensor(float32, size=(n, n))
      This is an initial guess for the dictionary of basis functions that
      we can use to descibe the images. n is the size of each image and also
      the size of the code
 all_params :
      --- MANDATORY ---
      'num_epochs': int
        The number of times to cycle over the whole dataset, reshuffling the
        order of the patches.
      'dictionary_update_algorithm' : str
        One of {'ica_natural_gradient'}
      'dict_update_param_schedule' : dictionary
        Dictionary containing iteration indexes at which to set/update
        parameters of the dictionary update algorithm. This will be algorithm
        specific. See the docstring for the respective algorithm in
        dictionary_learning/
      --- OPTIONAL ---
      'checkpoint_schedule' : dictionary, optional
        'checkpoint_folder_fullpath' tells us where to save the checkpoint
        files. All the other keys are specific iterations at which to save the
        parameters of the model (dictionary, codes, etc.) to disk. Values
        associated w/ each of these keys aren't used and can be set to None.
        We're just using the dictionary for its fast hash-based lookup.
      'training_visualization_schedule' : dictionary, optional
        'plot_object_reference' is a reference to a TrainingLivePlot object
        that we can call the UpdatePlot method on to display the progress of
        training the model. All the other keys are specific iterations at
        which to plot the dictionary and some sample codes. Again
  """

  ##########################
  # Setup and error checking
  ##########################
  assert 0 in all_params['dict_update_param_schedule']
  assert init_dictionary.size(0) == init_dictionary.size(1) # critically sampled
  # let's unpack all_params to make things a little less verbose...
  ### MANDATORY ###
  num_epochs = all_params['num_epochs']
  dict_update_alg = all_params['dictionary_update_algorithm']
  dict_update_param_schedule = all_params['dict_update_param_schedule']
  assert dict_update_alg in ['ica_natural_gradient']
  ### OPTIONAL ###
  if 'checkpoint_schedule' in all_params:
    ckpt_sched = all_params['checkpoint_schedule']
    ckpt_path = ckpt_sched.pop('checkpoint_folder_fullpath')
  else:
    ckpt_sched = None
  if 'training_visualization_schedule' in all_params:
    trn_vis_sched = all_params['training_visualization_schedule']
    lplot_obj_ref = trn_vis_sched.pop('liveplot_object_reference')
  else:
    trn_vis_sched = None

  if ckpt_sched is not None:
    # dump the parameters of this training session in human-readable JSON
    if not os.path.isdir(os.path.abspath(ckpt_path)):
      os.mkdir(ckpt_path)
    checkpointed_params = {
        k: all_params[k] for k in all_params if k not in
        ['checkpoint_schedule', 'training_visualization_schedule']}
    json.dump(checkpointed_params, open(ckpt_path+'/training_params.json', 'w'))

  # let's only import the things we need
  from analysis_transforms import invertible_linear
  if dict_update_alg == 'ica_natural_gradient':
    from dict_update_rules import ica_natural_gradient
  else:
    raise KeyError('Unrecognized dict update algorithm: ' + dict_update_alg)

  image_flatsize = image_dataset[0].shape[0]
  batch_size = image_dataset[0].shape[1]
  num_batches = len(image_dataset)
  ##################################
  # Done w/ setup and error checking
  ##################################

  dictionary = init_dictionary  # no copying, just a new reference

  starttime = time.time()
  total_iter_idx = 0
  for epoch_idx in range(num_epochs):
    for batch_idx, batch_images in enumerate(image_dataset):
      if total_iter_idx % 1000 == 0:
        print('Iteration', total_iter_idx, 'complete')
        print('Time elapsed', time.time() - starttime)

      if not batch_images.is_cuda:
        # We have to send image batch to the GPU
        batch_images.cuda(async=True)

      ####################
      # Run code inference
      ####################
      codes = invertible_linear.run(batch_images, dictionary)

      # check to see if we need to checkpoint the model or plot something
      if (ckpt_sched is not None and total_iter_idx in ckpt_sched):
        # In lieu of the torch-specific saver torch.save, we'll just use
        # pythons's standard serialization tool, pickle. That way we can mess
        # with the results without needing PyTorch.
        numpy_saved_dict = dictionary.cpu().numpy()
        pickle.dump(numpy_saved_dict, open(ckpt_path +
          '/checkpoint_dictionary_iter_' + str(total_iter_idx), 'wb'))
        numpy_saved_codes = codes.cpu().numpy()
        pickle.dump(numpy_saved_codes, open(ckpt_path +
          '/checkpoint_codes_iter_' + str(total_iter_idx), 'wb'))
      if (trn_vis_sched is not None and total_iter_idx in trn_vis_sched):
        for data_type in lplot_obj_ref.Requires():
          if data_type == 'dictionary':
            lplot_obj_ref.UpdatePlot(dictionary.cpu().numpy(), 'dictionary')
          elif data_type == 'codes':
            lplot_obj_ref.UpdatePlot(codes.cpu().numpy(), 'codes')

      #######################
      # Update the dictionary
      #######################
      # check to see if we need to set/update parameters
      if total_iter_idx in dict_update_param_schedule:
        d_upd_stp= dict_update_param_schedule[total_iter_idx]['stepsize']
        d_upd_niters = dict_update_param_schedule[total_iter_idx]['num_iters']
      if dict_update_alg == 'ica_natural_gradient':
        ica_natural_gradient.run(dictionary, codes, d_upd_stp, d_upd_niters)

      total_iter_idx += 1

    # we need to reshuffle the batches if we're not using a DataLoader
    if type(image_dataset) == torch.Tensor:
      # because of PyTorch's assumption of row-first reshaping this is a little
      # uglier than I would like...
      image_dataset = image_dataset.permute(0, 2, 1).reshape(
          -1, image_flatsize)[torch.randperm(num_batches * batch_size)].reshape(
              -1, batch_size, image_flatsize).permute(0, 2, 1)

    print("Epoch", epoch_idx, "finished")
    # let's make sure we release any unreferenced tensor to make their memory
    # visible to the OS
    torch.cuda.empty_cache()
