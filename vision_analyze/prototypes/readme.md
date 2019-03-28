# learning prototypes

- the idea here was to study and learn prototypes in a manner similar to [this paper](https://arxiv.org/abs/1806.10574)
    - the general idea, is to learn "prototypes" of classes and then compare test data points to these prototypes (in final activation space) to come up with an interpretation
- my overall takeaway from working with this style of interpretation was that it yielded **nicer looking, but not necessarily more accurate** interpretations
- differences between my method and the paper above
    - siamese setup - no need to project prototypes
    - instaed of L2 distance, used cosine similarity (maybe also use [lower dim Lk norm](https://bib.dbvis.de/uploadedFiles/155.pdf))

## observations
  - when initializing the prototypes to uniform noise, harder to train than when initializing to points
  - dot similarity is easier to achieve high acc than cosine sim
  - templates can be quite noisy, only in simple cases do they correctly learn sensible templates

## some background
- background
  - [cosine distances in dnns](https://arxiv.org/pdf/1702.05870.pdf) (there is also batch, layer, and weight normalization)
  - [prototypes for few-shot learning](https://arxiv.org/pdf/1703.05175v2.pdf)

## things left to do
- find modes in sampling setting
    - picking between data points as prototypes….
    - stability of learned prototypes?
- future work
   - use small spatial parts - e.g. bagnet
    - composing parts - smth like decision tree at top
    - reg input - tv, match lower lays, gan...
    - interpolation - adversarial training on prototypes, smooth function (e.g. [irevnet](https://github.com/jhjacobsen/pytorch-i-revnet), [nice](https://arxiv.org/abs/1410.8516), [invertible resnets](https://arxiv.org/pdf/1811.00995.pdf?fbclid=IwAR2O7cLHsrHRlI4i_qk22POUSzuMCDnzZjJsrDeJRE_gRgnlpAVVm-d5t2U)), force points to lie in space between the vectors
 - make prototypes nicer
   - better mnist gan
   - make loss 1, -1?
   - compare w/ l2 distance…
- see what prototypes are learned by imagenet models, neuroscience models