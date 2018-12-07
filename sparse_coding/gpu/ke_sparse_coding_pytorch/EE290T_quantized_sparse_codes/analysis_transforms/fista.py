"""
Implementation of Fast Iterative Soft Thresholding (Accelerated proximal grad)
"""
import torch

def run(images, dictionary, sparsity_weight, max_num_iters,
        convergence_epsilon=1e-3, nonnegative_only=False):
  """
  Runs steps of Fast Iterative Soft Thresholding, with fixed stepsize

  Termination is at the sooner of 1) code changes by less then
  convergence_epsilon, (per component, normalized by stepsize, on average)
  or 2) max_num_iters have been taken.

  Parameters
  ----------
  images : torch.Tensor(float32, size=(n, b))
      An array of images (probably just small patches) that to find the sparse
      code for. n is the size of each image and b is the number of images in
      this batch
  dictionary : torch.Tensor(float32, size=(n, s))
      This is the dictionary of basis functions that we can use to descibe the
      images. n is the size of each image and s in the size of the code.
  sparsity_weight : torch.Tensor(float32)
      This is the weight on the sparsity cost term in the sparse coding cost
      function. It is often denoted as \lambda
  max_num_iters : int
      Maximum number of steps of ISTA to run
  convergence_epsilon : float, optional
      Terminate if code changes by less than this amount per component,
      normalized by stepsize. Default 1e-3.
  nonnegative_only : bool, optional
      If true, our code values can only be nonnegative. We just chop off the
      left half of the FISTA soft thresholding function and it becomes a
      shifted RELU function. The amount of the shift from a generic RELU is
      precisely the sparsity_weight. Default False

  Returns
  -------
  codes : torch.Tensor(float32, size=(s, b))
      The set of codes for this set of images. s is the code size and b in the
      batch size.
  """
  # Stepsize set by the largest eigenvalue of the Gram matrix. Since this is
  # of size (s, s), and s >= n, we want to use the covariance matrix
  # because it will be of size (n, n) and have the same eigenvalues
  # ** For LARGE values of d = min(s, n), this will become a computational
  #    bottleneck. Consider setting lipschitz constant based on the
  #    backtracking rule outlined in Beck and Teboulle, 2009.
  lipschitz_constant = torch.symeig(
      torch.mm(dictionary, dictionary.t()))[0][-1]
  stepsize = 1. / lipschitz_constant
  t_kplusone = 1.

  codes = images.new_zeros(dictionary.size(1), images.size(1))
  old_codes = codes.clone()
  aux_points = codes.clone()
  #^ these are the auxiliary points that we actually compute gradients w.r.t.
  avg_per_component_change = torch.mean(torch.abs(codes - old_codes))

  iter_idx = 0
  while (iter_idx < max_num_iters and
         (avg_per_component_change > convergence_epsilon or iter_idx == 0)):
    old_codes = codes.clone()
    t_k = t_kplusone

    # gradient of l2 term is <dictionary^T, (<dictionary, aux> - images)>
    codes = aux_points - stepsize * torch.mm(
        dictionary.t(), torch.mm(dictionary, aux_points) - images)
    #^ pre-threshold values p - lambda*A^T(Ap - y)
    if nonnegative_only:
      codes.sub_(sparsity_weight * stepsize).clamp_(min=0.)
      #^ shifted rectified linear activation
    else:
      pre_threshold_sign = torch.sign(codes)
      codes.abs_()
      codes.sub_(sparsity_weight * stepsize).clamp_(min=0.)
      codes.mul_(pre_threshold_sign)
      #^ now contains the "soft thresholded" (non-rectified) output x_{k+1}

    t_kplusone = (1 + (1 + (4 * t_k**2))**0.5) / 2
    beta_kplusone = (t_k - 1) / t_kplusone

    change_in_codes = codes - old_codes

    aux_points = codes + beta_kplusone*(change_in_codes)

    avg_per_component_change = torch.mean(torch.abs(change_in_codes) / stepsize)
    iter_idx += 1

  return codes
