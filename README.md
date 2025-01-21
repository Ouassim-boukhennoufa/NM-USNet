# NM-USNet: A Novel Generative Model for Parathyroid Glands Detection in Nuclear Medicine
[NM-USNet Architecture](Images/Methodologie.pdf)
This repository contains different codes of a generative model **NM-USNet** that is a combination of a **Siamese Network** and a **U-Net** model for parathyroid glands detection through automatic image subtraction using 123-I and 99m-Tc images. The combination of the two components is monitored with an optimized loss function that includes reconstruction error (MSE of the U-NET) and similarity computed by the SN, with dynamic weighting adjustment methodology.

---

## Needs to know

- **Siamese Network**: Learns to compute similarity scores between image pairs using a shared feature extraction architecture. It needs to be trained before pair of real images and correspending noise.
- **U-Net Architecture**: Performs image-to-image mapping for reconstruction tasks, with an encoder-decoder architecture.
- **Combined Loss**: Integrates U-Net loss (MSE) and Siamese Network loss, dynamically weighted by `alpha` and `beta`, weights of the **U-Net** and **Siamese Network**, respectively.
- **10-Fold Cross-Validation**: To assess the performance and robustness.
- **Custom Metrics**: Includes Pearson correlation as a validation metric.

---


## Install dependencies using:
```bash
pip install -r requirements.txt
```
---
## Example execution of NM-USNet
### Due to data privacy, this example image is generated using XCAT simind simulator
![NM-USNet Example execution](Images/ExampleGeneration.png)
### A sample data of the synthetic ones are available in the folder Images, in numpy formats.
---
## An open science project regarding NM-USNet is available via this link [NM-USNet](https://dx.doi.org/doi:10.25666/DATAUBFC-2024-12-17)

---

## Cite this article
```bibtex
@article{BOUKHENNOUFA2025107493,
	title = {NM-USNet: A novel generative model for parathyroid glands detection in nuclear medicine},
	journal = {Biomedical Signal Processing and Control},
	volume = {104},
	pages = {107493},
	year = {2025},
	issn = {1746-8094},
	doi = {https://doi.org/10.1016/j.bspc.2025.107493},
	url = {https://www.sciencedirect.com/science/article/pii/S1746809425000047},
	author = {Ouassim Boukhennoufa and Laurent Comas and Jean-Marc Nicod and Noureddine Zerhouni and Hatem Boulahdour},}
```
