# Fully Convolutional Neural Network for Rapid Flood Segmentation in Synthetic Aperture Radar Imagery


<p align="center">
  <img width="500" height="400" src="https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/figures/Graphical_Abstract.png">
</p>


## Backgrouund

Rapid response to natural hazards, such as floods, is essential to mitigate loss of life and the reduction of suffering. For emergency response teams, access to timely and accurate data is essential. Satellite imagery offers a rich source of information which can be analysed to help determine regions affected by a disaster.  Much remote sensing flood analysis is semi-automated, with time consuming manual components requiring hours to complete. In this study, we present a fully automated approach to the rapid flood mapping currently carried out by many non-governmental, national and international organisations. We take a Convolutional Neural Network (CNN) based approach which isolates the flooded pixels in freely available Copernicus Sentinel-1 Synthetic Aperture Radar (SAR) imagery, requiring no optical bands and minimal pre-processing. We test a variety of CNN architectures and train our models on flood masks generated using a combination of classical semi-automated techniques and extensive manual cleaning and visual inspection. Our methodology reduces the time required to develop a flood map by 80%, while achieving strong performance over a wide range of locations and environmental conditions. Given the open-source data and the minimal image cleaning required, this methodology can also be integrated into end-to-end pipelines for more timely and continuous flood monitoring.

## Paper 

This GitHub repository contains the machine learning models described in Nemni, E.; Bullock, J.; Belabbes, S.; Bromley, L. Fully Convolutional Neural Network for Rapid Flood Segmentation in Synthetic Aperture Radar Imagery. Remote Sens. 2020, 12, 2532. https://doi.org/10.3390/rs12162532. Please see detailed information on the paper [here](paper.md).

## Dataset

The UNOSAT Flood Dataset has been created for this study using Copernicus Sentinel-1 satellite imagery acquired in Interferometric Wide Swath (IW) and provided as Level-1 Ground Range Detected (GRD)  products at a resolution of 10 m x 10 m with corresponding flood vectors stored in shapefile format.  The image name of each image used are listed in Table A1 in Appendix A of the paper. The analyses that acted as ground truth can be downloaded from the [here](https://cernbox.cern.ch/s/2PIXlW3TP2bP5LM). 

## Setup

Please see detailed setup instructions [here](docs/setup.md).

## Prepare the data 

1. Create an account on Alaska Satellite 
1. Run the [downloader Jupyter Notebook](downloader.ipynb) to dowload the Sentinel-1 data
1. Downlaod the groundtruth from [here](https://cernbox.cern.ch/s/2PIXlW3TP2bP5LM)


## Using the model

Please see detailed instructions [here](docs/flood_mapping_instructions.md).

## Acknowledgments

Please cite the paper as: 

```
@article{UNOSAT-FloodAI,
	title={Fully Convolutional Neural Network for Rapid Flood Segmentation in Synthetic Aperture Radar Imagery},
	author={Nemni, E.; Bullock, J.; Belabbes, S.; Bromley L.},
	journal={Remote Sensing},
	volume={12},
	number={8},
	article-number={2532},
	year={2020},
	month={12},
	day={},
	publisher={},
    url={},
    issn={},
    doi={https://doi.org/10.3390/rs12162532}
}


This software was developed in collaboration with UN Global Pulse and CERN Openlab. UN Global Pulse is the Secretary-General’s Innovation Lab — a hub for experimentation to support and advance the UN Charter. CERN Openlab is a public-private partnership through which CERN collaborates with leading ICT companies and other research organisations. 
