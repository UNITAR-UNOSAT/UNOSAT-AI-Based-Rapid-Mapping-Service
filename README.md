# Fully Convolutional Neural Network for Rapid Flood Segmentation in Synthetic Aperture Radar Imagery


<p align="center">
  <img width="500" height="400" src="https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/figures/Graphical_Abstract.png">
</p>


This GitHub repository contains the machine learning models described in Edoardo Nemnni, Joseph Bullock, Samir Belabbes, Lars Bromley Fully Convolutional Neural Network for Rapid Flood Segmentation in Synthetic Aperture Radar Imagery.

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

# Dataset

The UNOSAT Flood Dataset has been created for this study using Copernicus Sentinel-1 satellite imagery acquired in Interferometric Wide Swath (IW) and provided as Level-1 Ground Range Detected (GRD)  products at a resolution of 10 m x 10 m with corresponding flood vectors stored in shapefile format. Each image was downloaded from the [Copernicus Open Access Hub](https://scihub.copernicus.eu/dhus/#/home) in order to work on free and open-source data. The image name of each image used are listed in Table A1 in Appendix A of the paper. The analyses can be downloaded from the [UNOSAT Flood Portal](http://floods.unosat.org/geoportal/catalog/main/home.page), and the [UNOSAT page on the Humanitarian Data Exchange](https://data.humdata.org/organization/un-operational-satellite-appplications-programme-unosat).

Once the data has been downloaded, we compress it to 8-bit for computational efficiency which can be done as follows:

```
gdal_translate -co "COMPRESS=JPEG" -ot Byte -b 1 uncompressed.tif compressed.tif
```

After this, the image and its corresponding labels must be tiled. We use Python's math, numpy and opencv libraries to split the images and labels into 256x256 pixel tiles for training. An example of how to do this is the following:

```
image = cv.imread(image_path, -1)
label = cv.imread(label_path, -1)

tile_size = (256, 256)
offset = (256, 256)

flag=0 #flag value that corresponds to the pixel value in the frame 
count=0
for i in tqdm(range(int(math.ceil(image.shape[0]/(offset[1] * 1.0))))):
    for j in range(int(math.ceil(image.shape[1]/(offset[0] * 1.0)))):
        cropped_img = image[offset[1]*i:min(offset[1]*i+tile_size[1], image.shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], image.shape[1])]
        cropped_lab = label[offset[1]*i:min(offset[1]*i+tile_size[1], label.shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], label.shape[1])]
        #Save the tile only if none of the pixels has the value 'flag' and if at least one flood pixels is present
        if np.sum(cropped_img==flag) == 0 and np.sum(cropped_lab)>0: 
            count=count+1
        # Debugging the tiles
            #cv.imwrite(tile_img + save_name + str(i) + "_" + str(j) + ".png", cropped_img, [cv.IMWRITE_PNG_COMPRESSION, 0])
            #cv.imwrite(tile_lab + save_name + str(i) + "_" + str(j) + ".png", cropped_lab, [cv.IMWRITE_PNG_COMPRESSION, 0])
```

Alternative tiling mechanisms can be used depending on the overlap and zoom levels required.

# Examples

## XNet and U-Net Training
```bash
cd naive_segmentation
python model_training.py --config_file /configs/config_example.yaml
```
## XNet and U-Net Inference
```bash
cd naive_segmentation
python model_inference.py --config_file /configs/config_example.yaml
```
## Fastai Training and Inference

see notebook https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/Fastai%20training.ipynb

# Paper

## Abstract

Rapid response to natural hazards, such as floods, is essential to mitigate loss of life1and the reduction of suffering. For emergency response teams, access to timely and accurate data is essential. Satellite imagery offers a rich source of information which can be analysed to help determine regions affected by a disaster. Much remote sensing flood analysis is semi-automated, with time consuming manual components requiring hours to complete. In this study, we present a fully automated approach to the rapid flood mapping currently carried out by many non-governmental, national  and  international  organisations. We take a  Convolutional  Neural  Network  (CNN) based  approach  which  isolates  the  flooded  pixels  in  freely  available  Copernicus  Sentinel-1 Synthetic Aperture Radar (SAR) imagery, requiring no optical bands and minimal pre-processing. We test a variety of CNN architectures and train our models on flood masks generated using a combination of classical semi-automated techniques and extensive manual cleaning and visual inspection.  Our methodology reduces the time required to develop a flood map by 80%, while achieving strong performance over a wide range of locations and environmental conditions. Given the open-source data and the minimal image cleaning required, this methodology can also be integrated into end-to-end pipelines for more timely and continuous flood monitoring.

## Tables and Figures

Hyper-parameter tuning experimental setups for different architectures. Here we vary thebatch size, number of times the training dataset is passed to the model during training, the use of aweighted loss function, the filter depth of the convolutional layers (‘deep’ is four times the depth of‘shallow’) and the use of mixed precision during training. For the XNet and U-Net models this number refers to the number of times the entire training dataset was passed to themodel during training before Early Stopping was implemented. For the U-Net+ResNet case, the first number is the numberof times the dataset was passed when training just the head nodes of the architecture, and the latter the number of times itwas passed when training the entire network. In the latter case, at each epoch the validation loss was manualy comparedafter training and the best model selected.



| Name | Batch size | Dataset passes | Weighted Loss | Filter depth | Mixed |
| ------------- |:-------------:|:-----:|:-----:|:-----:|:-----:|
| XNet shallow   | 3 |1| | [16,32,64,128] | |
| XNet shallow weighted  | 3 |1| x | [16,32,64,128]| |
| XNet shallow weighted  b8| 8 |1| x | [16,32,64,128] | |
| XNet shallow weighted b8 d5 | 8 |5| x | [16,32,64,128] | |
| XNet shallow weighted | 3 |1| x | [56,128,256,512]| |
| UNet shallow   | 3 |1| | [16,32,64,128] | |
| UNet shallow weighted  | 3 |1| x | [16,32,64,128] | |
| UNet shallow weighted  d5| 3 |5| x | [16,32,64,128]| |
| UNet shallow weighted  b8| 8 |1| x |[16,32,64,128] | |
| UNet shallow weighted b8 d5 | 8 |5| x | [16,32,64,128]| |
| UNet shallow weighted | 3 |1| x | [56,128,256,512]| |
| UNet + resNet b4 d5/10 | 4 |5/10| x | [64,128,256,512,1204]| x |
| UNet + resNet b8 d5/10 | 8 |5/10| x | [64,128,256,512,1204]| x |
| UNet + resNet b8 d5/10 | 8 |5/10| x | [64,128,256,512,1204]|  |
| UNet + resNet b8 d10/20 | 8 |10/20| x | [64,128,256,512,1204]| x |
| UNet + resNet b32 d10/200 | 32 |10/20| x | [64,128,256,512,1204]| x |

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


| Model | Accuracy | Precision | Recall | Critical Sucess Index | F1 |
| ------------- |:-------------:| -----:| -----:|-----:| -----:|
| Baseline    | 91%| 62% | 84% | 0.55 | 0.71 |
| XNet    | 97%| 91% | 91% | 0.81 |0.91 |
| UNet    | 97%| 91% | 92% | 0.83 |0.91 |
| U-Net + ResNet   | 97%| 91% | 92% | 0.77| 0.92 |

Example outputs of the best performing U-Net+ResNet, after probability thresholding, can be seen beloe.  In particular, we see the neural network's ability to detect the flood area with minimal cleaning in comparison to the ground-truth data. The baseline (third column was generated using the automatic threshold-based method and would require significantly more noise reduction in post-processing.

<p align="center">
   <img src="https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/figures/results_plot_28.png" width="700" />
	
From left to right: tiles of different analysis are shown with the corresponding ground truth,baseline and neural network prediction.


Examples of well detected tiles from both the automatic histogram-based method and this neural network, particularly highlighting severe flooded regions:

<p align="center">
<img src="https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/figures/results_plot_34.png" width="700" /> 
</p>
