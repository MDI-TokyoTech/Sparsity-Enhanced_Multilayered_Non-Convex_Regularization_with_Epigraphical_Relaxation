# Sparsity-Enhanced Multilayered Non-Convex Regularization with Epigraphical Relaxation for Debiased Signal Recovery

This is demo programs for epigraphically-relaxed linearly-involved generalized Moreau-enhanced model (ER-LiGME model) proposed in the following reference:

A. Katsuma, S. Kyochi, S. Ono, I. Selesnick, "Sparsity-Enhanced Multilayered Non-Convex Regularization with Epigraphical Relaxation for Debiased Signal Recovery", 2024.

For more details, see the following

- Preprint paper: https://arxiv.org/abs/2409.14768

## How to use

**1) Prepare test images**
 - Place image files in the `/images` directory.
 - Example: [The Berkeley Segmentation Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)

**2) Setting parameters**
 - Set the path to target image files
 - Set the strength of noise
 - Adjust the regularization parameters
   - Details for each parameter are explained in the code comments and the referenced paper.

**3) Run demo_*.m**
 - image denoising: 
     demo_denoising.m
 - compressed image sensing reconstruction:
     demo_CSR.m, 
     demo_CSR_single.m (single precision)
 - principal component analysis of shifted signals:
     demo_FRPCA.m

## Citation
If you use this code, please cite the following paper:
```
@misc{katsuma2024sparsityenhancedmultilayerednonconvexregularization,
      title={Sparsity-Enhanced Multilayered Non-Convex Regularization with Epigraphical Relaxation for Debiased Signal Recovery}, 
      author={Akari Katsuma and Seisuke Kyochi and Shunsuke Ono and Ivan Selesnick},
      year={2024},
      eprint={2409.14768},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2409.14768}, 
}
```