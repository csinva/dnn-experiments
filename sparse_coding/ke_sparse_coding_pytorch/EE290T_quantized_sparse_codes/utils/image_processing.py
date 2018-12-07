"""
Some utilities for wrangling image data
"""

import numpy as np
import scipy.io

from matplotlib import pyplot as plt

def get_low_pass_filter(DFT_num_samples, filter_parameters):
  """
  Returns the DFT of a lowpass filter that can be applied to an image

  Parameters
  ----------
  DFT_num_samples : (int, int)
      The number of samples on the DFT vertical axis and the DFT horizontal axis
  filter_parameters : dictionary
      Parameters of the filter. These may vary depending on the filter shape.
      Smoother filters reduce ringing artifacts but can throw away a lot of
      information in the middle frequencies. For now we just have 'exponential'
      but may add other filter shapes in the future. The shape-specific filter
      parameters are given below
      'shape': 'exponential'
        'cutoff' : float \in [0, 1]
            A fraction of the 2d nyquist frequency at which to set the cutoff.
        'order' : float \in [1, np.inf)
            The order of the exponential. In the Vision Research paper this is
            4. We can make the cutoff sharper by increasing this number
  Returns
  -------
  lpf_DFT = ndarray(complex128, size(DFT_num_samples[0], DFT_num_samples[1]))
      The DFT of the low pass filter
  """
  nv = DFT_num_samples[0]
  nh = DFT_num_samples[1]

  if filter_parameters['shape'] == 'exponential':
    assert filter_parameters['cutoff'] <= 1.0
    assert filter_parameters['cutoff'] >= 0.0
    assert filter_parameters['order'] >= 1.0
    freqs_vert = np.arange(nv) * (1. / nv)
    # we want the filter to reflect the 'centering' operation of np.fft.fftshift
    if nv % 2 == 0:   # even number of samples
      freqs_vert[nv//2:] = freqs_vert[nv//2:] - 1
    else:
      freqs_vert[(nv//2)+1:] = freqs_vert[(nv//2)+1:] - 1
    freqs_vert = np.fft.fftshift(freqs_vert)

    # do this for the horizontal axis
    freqs_horz = np.arange(nh) * (1. / nh)
    # we want the filter to reflect the 'centering' operation of np.fft.fftshift
    if nh % 2 == 0:   # even number of samples
      freqs_horz[nh//2:] = freqs_horz[nh//2:] - 1
    else:
      freqs_horz[(nh//2)+1:] = freqs_horz[(nh//2)+1:] - 1
    freqs_horz = np.fft.fftshift(freqs_horz)

    two_d_freq = np.meshgrid(freqs_vert, freqs_horz, indexing='ij')
    spatial_freq_mag = (np.sqrt(np.square(two_d_freq[0]) +
                                np.square(two_d_freq[1])))
    lpf_DFT_mag = np.exp(
        -1. * np.power(spatial_freq_mag / (0.5 * filter_parameters['cutoff']),
                       filter_parameters['order']))
    #^ 0.5 is the 2d spatial nyquist frequency
    lpf_DFT_mag[lpf_DFT_mag < 1e-7] = 1e-7
    #^ avoid filter magnitudes that are 'too small' because this will make
    #  undoing the filter introduce arbitrary high-frequency noise
    lpf_DFT_phase = np.zeros(spatial_freq_mag.shape)

  return lpf_DFT_mag * np.exp(1j * lpf_DFT_phase)


def get_whitening_ramp_filter(DFT_num_samples):
  """
  Returns the DFT of a simple 'magnitude ramp' filter that whitens data

  Parameters
  ----------
  DFT_num_samples : (int, int)
      The number of samples in the DFT vertical axis and the DFT horizontal axis

  Returns
  -------
  wf_DFT = ndarray(complex128, size(DFT_num_samples[0], DFT_num_samples[1]))
      The DFT of the whitening filter
  """
  nv = DFT_num_samples[0]
  nh = DFT_num_samples[1]

  freqs_vert = np.arange(nv) * (1. / nv)
  # we want the filter to reflect the 'centering' operation of np.fft.fftshift
  if nv % 2 == 0:   # even number of samples
    freqs_vert[nv//2:] = freqs_vert[nv//2:] - 1
  else:
    freqs_vert[(nv//2)+1:] = freqs_vert[(nv//2)+1:] - 1
  freqs_vert = np.fft.fftshift(freqs_vert)

  # do this for the horizontal axis
  freqs_horz = np.arange(nh) * (1. / nh)
  # we want the filter to reflect the 'centering' operation of np.fft.fftshift
  if nh % 2 == 0:   # even number of samples
    freqs_horz[nh//2:] = freqs_horz[nh//2:] - 1
  else:
    freqs_horz[(nh//2)+1:] = freqs_horz[(nh//2)+1:] - 1
  freqs_horz = np.fft.fftshift(freqs_horz)

  two_d_freq = np.meshgrid(freqs_vert, freqs_horz, indexing='ij')
  spatial_freq_mag = (np.sqrt(np.square(two_d_freq[0]) +
                              np.square(two_d_freq[1])))
  wf_DFT_mag = spatial_freq_mag / np.max(spatial_freq_mag)
  wf_DFT_mag[wf_DFT_mag < 1e-7] = 1e-7
  #^ avoid filter magnitudes that are 'too small' because this will make
  #  undoing the filter introduce arbitrary high-frequency noise
  wf_DFT_phase = np.zeros(spatial_freq_mag.shape)

  return wf_DFT_mag * np.exp(1j * wf_DFT_phase)


def filter_image(image, filter_DFT):
  """
  Just takes the DFT of a filter and applies the filter to an image

  This may optionally pad the image so as to match the number of samples in the
  filter DFT. We should make sure this is greater than or equal to the size of
  the image.
  """
  assert image.dtype == 'float32'
  assert filter_DFT.shape[0] >= image.shape[0], "don't undersample DFT"
  assert filter_DFT.shape[1] >= image.shape[1], "don't undersample DFT"
  filtered_with_padding = np.real(
      np.fft.ifft2(np.fft.ifftshift(
        filter_DFT * np.fft.fftshift(np.fft.fft2(image, filter_DFT.shape))),
        filter_DFT.shape)).astype('float32')
  return filtered_with_padding[0:image.shape[0], 0:image.shape[1]]


def whiten_center_surround(image):
  """
  Applies the scheme described in the Vision Research sparse coding paper

  We have the composition of a low pass filter with a ramp in spatial frequency
  which together produces a center-surround filter in the image domain

  Parameters
  ----------
  image : ndarray(float32 or uint8, size=(h, w))
      An image of height h and width w
  """
  lpf = get_low_pass_filter(image.shape,
      {'shape': 'exponential', 'cutoff': 0.8, 'order': 4.0})
  wf = get_whitening_ramp_filter(image.shape)
  combined_filter = wf * lpf
  combined_filter /= np.max(np.abs(combined_filter))
  #^ make the maximum filter magnitude equal to 1
  return filter_image(image, combined_filter)


def center_on_origin(flat_data):
  """
  Makes each component of data have mean zero across the dataset

  Parameters
  ----------
  flat_data : ndarray(float32 or uint8, size=(n, D))
      n is the dimensionality of a single datapoint and D is the size of the
      dataset over which we are taking the mean.

  Returns
  -------
  centered_data : ndarray(float32, size=(n, D))
      The data, now with mean 0 in each component
  original_means : ndarray(float32, size=(n,))
      The componentwise means of the original data. Can be used to
      uncenter the data later (for instance, after dictionary learning)
  """
  assert flat_data.dtype in ['float32', 'uint8']
  original_means = np.mean(flat_data, axis=1)
  return (flat_data - original_means[:, None]).astype('float32'), original_means


def normalize_variance(flat_data):
  """
  Normalize each component to have a variance of 1 across the dataset

  Parameters
  ----------
  flat_data : ndarray(float32 or uint8, size=(n, D))
      n is the dimensionality of a single datapoint and D is the size of the
      dataset over which we are taking the variance.

  Returns
  -------
  normalized_data : ndarray(float32 size=(n, D))
      The data, now with variance
  original_variances : ndarray(float32, size=(n,))
      The componentwise variances of the original data. Can be used to
      unnormalize the data later (for instance, after dictionary learning)
  """
  assert flat_data.dtype in ['float32', 'uint8']
  original_variances = np.var(flat_data, axis=1)
  return ((flat_data / np.sqrt(original_variances)[:, None]).astype('float32'),
          original_variances)


def patches_from_single_image(image, patch_dimensions):
  """
  Extracts tiled patches from a single image

  Parameters
  ----------
  image : ndarray(float32 or uint8, size=(h, w))
      An image of height h and width w
  patch_dimensions : tuple(int, int)
      The size in pixels of each patch

  Returns
  -------
  patches : ndarray(float32 or uint8, size=(ph*pw, k))
      An array of flattened patches each of height ph and width pw. k is the
      number of total patches that were extracted from the full image
  patch_positions : list(tuple(int, int))
      The position in pixels of the upper-left-hand corner of each patch within
      the full image. Used to retile the full image after processing the patches
  """
  assert image.shape[0] / patch_dimensions[0] % 1 == 0
  assert image.shape[1] / patch_dimensions[1] % 1 == 0
  assert image.dtype in ['float32', 'uint8']

  num_patches_vert = image.shape[0] // patch_dimensions[0]
  num_patches_horz = image.shape[1] // patch_dimensions[1]
  patch_flatsize = patch_dimensions[0] * patch_dimensions[1]
  patch_positions = []  # keep track of where each patch belongs
  patches = np.zeros([patch_flatsize, num_patches_vert * num_patches_horz],
                     dtype=image.dtype)
  p_idx = 0
  for patch_idx_vert in range(num_patches_vert):
    for patch_idx_horz in range(num_patches_horz):
      pos_vert = patch_idx_vert * patch_dimensions[0]
      pos_horz = patch_idx_horz * patch_dimensions[1]
      patches[:, p_idx] = image[
          pos_vert:pos_vert+patch_dimensions[0],
          pos_horz:pos_horz+patch_dimensions[1]].reshape((patch_flatsize,))
      patch_positions.append((pos_vert, pos_horz))
      #^ upper left hand corner position in pixels in the original image
      p_idx += 1
  return patches, patch_positions


def assemble_image_from_patches(patches, patch_dimensions, patch_positions):
  """
  Tiles an image from patches

  Parameters
  ----------
  patches : ndarray(float32 or uint8, size=(ph*pw, k))
      An array of flattened patches each of height ph and width pw. k is the
      number of total patches that were extracted from the full image
  patch_dimensions : tuple(int, int)
      The size in pixels of each patch
  patch_positions : list(tuple(int, int))
      The position in pixels of the upper-left-hand corner of each patch within
      the full image.

  Returns
  -------
  image : ndarray(float32 or uint8, size=(h, w))
      An image of height h and width w
  """
  assert patches.dtype in ['float32', 'uint8']

  full_img_height = (np.max([x[0] for x in patch_positions]) +
                     patch_dimensions[0])
  full_img_width = (np.max([x[1] for x in patch_positions]) +
                    patch_dimensions[1])
  full_img = np.zeros([full_img_height, full_img_width], dtype=patches.dtype)
  for patch_idx in range(patches.shape[1]):
    vert = patch_positions[patch_idx][0]
    horz = patch_positions[patch_idx][1]
    full_img[vert:vert+patch_dimensions[0], horz:horz+patch_dimensions[1]] = \
        patches[:, patch_idx].reshape(patch_dimensions)

  return full_img


def create_patch_training_set(order_of_preproc_ops, patch_dimensions,
    batch_size, num_batches, edge_buffer, dataset, datasetparams):
  """
  Creates a large batch of training patches from one of our available datasets

  Parameters
  ----------
  order_of_preproc_ops : list(str)
      Specifies the preprocessing operations to perform on the data. Currently
      available operations are {'patch', 'center', 'normalize_variance',
      'whiten_center_surround'}.
      Example: (we want to perform Bruno's whitening in the fourier domain and
                also have the components of each patch have zero mean)
          ['whiten_center_surround', 'patch', 'zero_mean']
  patch_dimensions : tuple(int, int)
      The size in pixels of each patch
  batch_size : int
      The number of patches in a batch
  num_batches : int
      The total number of batches to assemble
  edge_buffer : int
      The buffer from the edge of the image from which we will not include any
      patches.
  dataset : str
      The name of the dataset to grab patches from. Currently one of
      {'Field_NW_unwhitened', 'Field_NW_whitened'}.
  datasetparams : dictionary
      A dictionary of parameters that may be specific to the dataset. Currently
      just specifies the filepath of the data file and which images to
      exclude from the training set.
      'filepath' : str
      'exclude' : list(int)

  Returns
  -------
  return_dict : dictionary
    'batched_patches' : ndarray(float32, size=(k, n, b))
        The patches training set where k=num_batches,
        n=patch_dimensions[0]*patch_dimensions[1], and b=batch_size. These are
        patches ready to be sent to the gpu and consumed in PyTorch
    'orignal_patch_means' : ndarray(float32, size=(n,)), optional
        If 'centering' was requested as a preprocessing step we return the
        original patch means so that future data can be processed using this
        same exact centering
    'orignal_patch_variances' : ndarray(float32, size=(n,)), optional
        If 'normalize_variance' was requested as a preprocessing step we return
        the original patch variances so that future data can be processed using
        this same exact normalization
  """
  if dataset == 'Field_NW_whitened':
    # data is stored as a .mat file
    unprocessed_images = scipy.io.loadmat(
      datasetparams['filepath'])['IMAGES'].astype('float32')
  elif dataset == 'Field_NW_unwhitened':
    unprocessed_images = scipy.io.loadmat(
      datasetparams['filepath'])['IMAGESr'].astype('float32')
  else:
    raise KeyError('Unrecognized dataset ' + dataset)

  already_patched_flag = False
  p_imgs = np.copy(unprocessed_images)
  for preproc_op in order_of_preproc_ops:

    if preproc_op == 'whiten_center_surround':
      for img_idx in range(p_imgs.shape[2]):
        p_imgs[:, :, img_idx] = whiten_center_surround(p_imgs[:, :, img_idx])

    elif preproc_op == 'patch':
      max_vert_pos = p_imgs.shape[0] - patch_dimensions[0] - edge_buffer
      min_vert_pos = edge_buffer
      max_horz_pos = p_imgs.shape[1] - patch_dimensions[1] - edge_buffer
      min_horz_pos = edge_buffer
      eligible_image_inds = np.array([x for x in range(p_imgs.shape[2]) if
                                      x not in datasetparams['exclude']])

      all_patches = np.zeros(
        [patch_dimensions[0]*patch_dimensions[1], num_batches*batch_size],
        dtype='float32')

      p_idx = 0
      for batch_idx in range(num_batches):
        if batch_idx % 1000 == 0:
          print('Finished creating', batch_idx, 'batches')
        for _ in range(batch_size):
          vert_pos = np.random.randint(low=min_vert_pos, high=max_vert_pos)
          horz_pos = np.random.randint(low=min_horz_pos, high=max_horz_pos)
          img_idx = np.random.choice(eligible_image_inds)
          all_patches[:, p_idx] = p_imgs[
            vert_pos:vert_pos+patch_dimensions[0],
            horz_pos:horz_pos+patch_dimensions[1],
            img_idx].reshape([patch_dimensions[0]*patch_dimensions[1]])
          p_idx += 1
      print('Done.')
      already_patched_flag = True

    elif preproc_op == 'center':
      if not already_patched_flag:
        raise KeyError('You ought to patch the data before trying to center it')
      all_patches, orig_means = center_on_origin(all_patches)

    elif preproc_op == 'normalize_variance':
      if not already_patched_flag:
        raise KeyError('You ought to patch the data before normalizing it')
      all_patches, orig_variances = normalize_variance(all_patches)

    else:
      raise KeyError('Unrecognized preprocessing op ' + preproc_op)

  # now we finally chunk this up into batches and return
  return_dict = {'batched_patches': all_patches.T.reshape(
                    (num_batches, batch_size, -1)).transpose((0, 2, 1))}
                 #^ size=(k, n, b)
  if 'center' in order_of_preproc_ops:
    return_dict['original_patch_means'] = orig_means
  if 'normalize_variance' in order_of_preproc_ops:
    return_dict['original_patch_variances'] = orig_variances

  return return_dict
