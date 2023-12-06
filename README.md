# DAPIArtifactRemoval

The repository is for the paper 'MITIGATING OVER SATURATED FLUORESCENCE IMAGE THROUGH SEMI-SUPERVISED GENERATIVE ADVERSARIAL NETWORK.'

The proposed framework leverages:

1. SynsegNet (https://github.com/MASILab/SynSeg-Net) - stage 1 for training to generate a large volume of pseudo-paired data.
2. pix2pixHD (https://github.com/NVIDIA/pix2pixHD) - stage 2 for training high-resolution image translation, including pseudo and real paired data.
3. DeepCell mesmer segmentation (https://github.com/vanvalenlab/deepcell-applications).
4. The validation is about nuclei instance segmentation using MEDIAR (https://github.com/Lee-Gihun/MEDIAR), as the model is trained on a heterogeneous data cohort with different magnifications and micron sizes. 

We currently upload a few training scripts, tips and configurations. 
