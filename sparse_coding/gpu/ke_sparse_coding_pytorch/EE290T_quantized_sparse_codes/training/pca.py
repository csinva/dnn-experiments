"""
This just directly computes the PCA dictionary
"""

import numpy as np
import torch

def train_dictionary(image_dataset):
  """
  Compute the PCA dictionary matrix in one step

  Parameters
  ----------
  image_dataset : torch.Tensor(float32, size=(n, D))
      The full dataset, patches packed one into each column. Each row of
      this matrix should be mean zero.

  Returns
  -------
  PCA_dictionary : torch.Tensor(float32, size(n, s))
      Each basis element is a column of this matrix. The original data dimension
      is n.
  """
  assert np.all(np.abs(np.zeros((image_dataset.shape[0],), dtype='float32') -
                       torch.mean(image_dataset, dim=1).cpu().numpy()) < 1e-6)

  if image_dataset.size(0) > image_dataset.size(1):
    # If the dimensionality of each datapoint is high, we want to compute the
    # SVD of the data directly to avoid forming a huge covariance matrix
    U, s, _ = torch.svd(image_dataset)
  else:
    # we'll actually compute the covariance matrix because it will be relatively
    # smaller than the dataset matrix. But we'll still use the SVD to get the
    # PCA bases because it's more numerically stable then eig
    covar = torch.mm(image_dataset, image_dataset.t()) / image_dataset.size(1)
    U, w, _ = torch.svd(covar)

  # remember the PCA transform not completely unique. The defining criteria is
  # *invariant to sign flips of any of the basis vectors
  return U.float()
