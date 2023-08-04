# Setup Instructions

Last update: May 31, 2023

## Linux Setup Instructions

The model runs in a Linux environment. It has been tested with Ubuntu 18.04.4 or higher, Python 3.7.6 or higher, CUDA 10.0 or higher. Follow these steps to set up your Linux workstation.

### Clone git repo

```
$ git clone https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service.git
```

### Set up virtualenv

```
$ python -m venv flood_mapping_env
$ pip install --upgrade pip
$ cd <source-directory>
$ pip install -r requirements.txt
$ conda install anaconda gdal
```

### Troubleshooting

**Error: GetProjection() RuntimeError: PROJ: proj_create_from_database: Cannot find proj.db.**

Check https://github.com/OSGeo/gdal/issues/2248 or https://blog.csdn.net/qq_31793023/article/details/103622134 and set and Environment Variable PROJ_LIB with all the path where you find proj.db. 

```bash
 find / -type d -name 'proj'
 set PROJ_LIB /root/anaconda3/pkgs/proj-6.2.1-hc80f0dc_0/share/proj
```

**Error: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.**

Check https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html
