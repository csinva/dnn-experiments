"""
This updates the dictionary according to the ICA natural gradient
"""
import torch

def run(dictionary, codes, stepsize=0.001, num_iters=1):
  """
  Updates the dictionary according to the ICA natural gradient learning rule

  Parameters
  ----------
  dictionary : torch.Tensor(float32, size=(n, s))
      This is the dictionary of basis functions that we can use to descibe the
      images. n is the size of each image and s in the size of the code.
  codes : torch.Tensor(float32, size=(s, b))
      This is the current set of codes for a batch of images. s is the
      dimensionality of the code and b is the number of images in the batch
  stepsize : torch.Tensor(float32)
      The step size for each iteration of natural gradient. Keep this small
  num_iters : int
      Number of steps of natural gradient update to run
  """
  for iter_idx in range(num_iters):
    # the update for a single patch is A(zs^T - I) where s is the code, z is
    # the sign of the code, and I is the identity matrix. We can average this
    # update over a batch of images which is what we do below:
    dict_update = stepsize * (
        torch.mm(dictionary, torch.mm(torch.sign(codes), codes.t()) /
                             codes.size(1)) - dictionary)
    dictionary.add_(dict_update)  # we want to *ascend* the gradient
