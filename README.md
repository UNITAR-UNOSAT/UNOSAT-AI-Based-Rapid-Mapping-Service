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

## Installation

Use the package manager [anaconda](https://www.anaconda.com/) to install the virtual environment.

```bash
bash Anaconda3-2020.02-Linux-x86_64.sh
conda create -n flood_mapping 
source activate flood_mapping
pip install -r requirement.txt
conda install anaconda gdal
```
### Possible error and solution

Error: GetProjection() RuntimeError: PROJ: proj_create_from_database: Cannot find proj.db. 
If so check https://github.com/OSGeo/gdal/issues/2248 or https://blog.csdn.net/qq_31793023/article/details/103622134 and set and Environment Variable PROJ_LIB with all the path where you find proj.db. 

```bash
 find / -type d -name 'proj'
 set PROJ_LIB /root/anaconda3/pkgs/proj-6.2.1-hc80f0dc_0/share/proj
```


