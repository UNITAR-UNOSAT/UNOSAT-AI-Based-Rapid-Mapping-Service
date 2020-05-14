# Rapid Flood Segmentation in SAR imagery based on Fully Convolutional Neural Network

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
## Requirments

Ubuntu ? or higher
Python 3.7.6 or higher
CUDA 10.0 or higher
pytorch ? or higher

## Installation

Use the package manager [anaconda](https://www.anaconda.com/) to install the virtual environment.

```bash
bash Anaconda3-2020.02-Linux-x86_64.sh
conda create -n flood_mapping 
source activate flood_mapping
pip install -r requirement.txt
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
