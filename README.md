# Rapid Flood Segmentation in SAR imagery based on Fully Convolutional Neural Network


<p align="center">
  <img width="400" height="500" src="https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/figures/Graphical_Abstract.png">
</p>


This GitHub repository contains the machine learning models described in Edoardo Nemnni, Joseph Bullock, Samir Belabbes, Lars Bromley (Rapid) Flood Segmentation in SAR imagery based onFully Convolutional Neural Network.

```
@article{,
	title={},
	author={},
	journal={Remote Sensing},
	volume={12},
	number={8},
	article-number={},
	year={2020},
	month={},
	day={},
	publisher={},
    url={},
    issn={},
    doi={}
}
```
# Instructions

## Requirments

Ubuntu 18.04.4 or higher

Python 3.7.6 or higher

CUDA 10.0 or higher


## Installation



Use the package manager [anaconda](https://www.anaconda.com/) to install the virtual environment.

```bash
bash Anaconda3-2020.02-Linux-x86_64.sh
conda create -n flood_mapping 
source activate flood_mapping
pip install fastai keras tensorflow 
pip install yaml pandas tqdm opencv-python
conda install anaconda gdal
```
### Troubleshooting

Error: GetProjection() RuntimeError: PROJ: proj_create_from_database: Cannot find proj.db. 
If so check https://github.com/OSGeo/gdal/issues/2248 or https://blog.csdn.net/qq_31793023/article/details/103622134 and set and Environment Variable PROJ_LIB with all the path where you find proj.db. 

```bash
 find / -type d -name 'proj'
 set PROJ_LIB /root/anaconda3/pkgs/proj-6.2.1-hc80f0dc_0/share/proj
```


Error: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.

Solution: https://docs.fast.ai/troubleshoot.html#initial-installation + https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html

# Examples

# Paper

## Abstract

Rapid response to natural hazards, such as floods, is essential to mitigate loss of life and the reduction of suffering. For emergency response teams, access to timely and accurate data is essential. Satellite imagery offers a rich source of information which can be analysed to help determine regions affected by a disaster.  Much remote sensing flood analysis is semi-automated, with time consuming manual components requiring hours to complete. In this study, we present a fully automated approach to the rapid flood mapping currently carried out by many national and international organisations. We take a Convolutional Neural Network based approach which isolates the flooded pixels in freely available Synthetic Aperture Radar (SAR) imagery, requiring no optical bands and minimal cleaning. Our methodology reduces the time required to develop a flood map by 80%, while maintaining the required performance standards - reaching 97% accuracy compared to human annotated maps. Given the open-source data and the minimal image cleaning required, this methodology can also be integrated into end-to-end pipelines for more timely and continuous flood monitoring.

## Tables and Figures

Hyper-parameter tuning experimental setups for different architectures. Here we vary thebatch size, number of times the training dataset is passed to the model during training2, the use of aweighted loss function, the filter depth of the convolutional layers (‘deep’ is four times the depth of‘shallow’) and the use of mixed precision during training. For the XNet and U-Net models this number refers to the number of times the entire training dataset was passed to themodel during training before Early Stopping was implemented. For the U-Net+ResNet case, the first number is the numberof times the dataset was passed when training just the head nodes of the architecture, and the latter the number of times itwas passed when training the entire network. In the latter case, at each epoch the validation loss was manualy comparedafter training and the best model selected.



| Name | Batch size | Dataset passes | Weighted Loss | Filter depth | Mixed |
| ------------- |:-------------:|:-----:|:-----:|:-----:|:-----:|
| XNet shallow   | 3 |1| | shallow | |
| XNet shallow weighted  | 3 |1| x | shallow | |
| XNet shallow weighted  b8| 8 |1| x | shallow | |
| XNet shallow weighted b8 d5 | 8 |5| x | shallow | |
| XNet shallow weighted | 3 |1| x | deep| |
| UNet shallow   | 3 |1| | shallow | |
| UNet shallow weighted  | 3 |1| x | shallow | |
| UNet shallow weighted  d5| 3 |5| x | shallow | |
| UNet shallow weighted  b8| 8 |1| x | shallow | |
| UNet shallow weighted b8 d5 | 8 |5| x | shallow | |
| UNet shallow weighted | 3 |1| x | deep| |
| UNet + resNet b4 d5/10 | 4 |5/10| x | default| x |
| UNet + resNet b8 d5/10 | 8 |5/10| x | default| x |
| UNet + resNet b8 d5/10 | 8 |5/10| x | default|  |
| UNet + resNet b8 d10/20 | 8 |10/20| x | default| x |
| UNet + resNet b32 d10/200 | 32 |10/20| x | default| x |

Precision-recall  curved  for  different  experiments from top to bottom. XNet  trials;  U-Net  trials; and U-Net+ResNet trials. The curves show the precision and recall values at different probability thresholdsranging from 0.01 to 0.99.

<p align="center">
   <img src="https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/figures/xnet_all_pr_curves.png" width="500" /> 
</p>

<p align="center">
   <img src="https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/figures/unet_all_pr_curves.png" width="500" /> 
</p>

<p align="center">
   <img src="https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/figures/fastai_unet_all_pr_curves.png" width="500" /> 
</p>

Overall quantitative comparison of  the  best  XNet,   U-Net  and  U-Net+ResNet  models  after hyper-parameter tuning compared against the automatic histogram based method baseline result.


| Model | Accuracy | Precision | Recall | F1 |
| ------------- |:-------------:| -----:| -----:| -----:|
| Baseline    | 91%| 63% | 84% | 0.72 |
| XNet    | 97%| 91% | 91% | 0.91 |
| UNet    | 97%| 91% | 92% | 0.91 |
| U-Net + ResNet   | 97%| 91% | 92% | 0.92 |

Example outputs of the best performing U-Net+ResNet, after probability thresholding, can be seen beloe.  In particular, we see the neural network's ability to detect the flood area with minimal cleaning in comparison to the ground-truth data. The baseline (third column was generated using the automatic threshold-based method and would require significantly more noise reduction in post-processing.

<p align="center">
   <img src="https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/figures/results_plot_28.png" width="700" />
	
From left to right: tiles of different analysis are shown with the corresponding ground truth,baseline and neural network prediction.


Eexamples of well detected tiles from both the automatic histogram-based method and this neural network, particularly highlighting severe flooded regions:

<p align="center">
<img src="https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/figures/results_plot_34.png" width="700" /> 
</p>
