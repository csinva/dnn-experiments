"""
Simply implements a SC steepest descent update to the dictionary elements
"""
import torch

def run(images, dictionary, codes, stepsize=0.001, num_iters=1,
        normalize_dictionary=True):
  """
  Runs num_iters steps of SC steepest descent on the dictionary elements

  Parameters
  ----------
  images : torch.Tensor(float32, size=(n, b))
      An array of images (probably just small patches) that to find the sparse
      code for. n is the size of each image and b is the number of images in
      this batch
  dictionary : torch.Tensor(float32, size=(n, s))
      This is the dictionary of basis functions that we can use to descibe the
      images. n is the size of each image and s in the size of the code.
  codes : torch.Tensor(float32, size=(s, b))
      This is the current set of codes for a batch of images. s is the
      dimensionality of the code and b is the number of images in the batch
  stepsize : torch.Tensor(float32)
      The step size for each iteration of steepest descent. Keep this small
  num_iters : int
      Number of steps of steepest descent to run
  normalize_dictionary : bool, optional
      If true, we normalize each dictionary element to have l2 norm equal to 1
      before we return.
  """
  for iter_idx in range(num_iters):
    dictionary.sub_(stepsize * torch.mm(torch.mm(dictionary, codes) - images,
                                        codes.t()) / codes.size(1))
    if normalize_dictionary:
      dictionary.div_(dictionary.norm(p=2, dim=0))
