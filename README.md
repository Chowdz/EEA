# Satellite Image Inpainting based with Edge Enhanced Attention
This repository contains the implementation of the research paper "Satellite Image Inpainting based with Edge Enhanced Attention". The paper is currently under submission.

## Introduction
Satellite image inpainting is an important task in remote sensing and image processing. This project implements a novel inpainting method that leverages edge-enhanced attention mechanisms base on a pure Transformer architecture to improve the quality and accuracy of inpainted satellite images.

## Requirements
Python >=3.6

PyTorch >=1.6

NVIDIA GPU + CUDA cuDNN

## Installation
1. Clone the repository:
   
   ```git clone https://github.com/your-username/satellite-image-inpainting.git```

  ```cd satellite-image-inpainting```
  
2. Install dependencies:
   
  ```pip install -r requirements.txt```
  
## Dataset

We utilized [Google Earth Engine (GEE) API](https://doi.org/10.1016/j.rse.2017.06.031) to obtain a dataset of satellite images collected before May 31, 2003. Each image has dimensions of 8131×7061 pixels, a spatial resolution of 30 meters, and consists of 8 bands. The dataset was curated to include satellite images with cloud coverage below 1%. Our experiments focused specifically on bands B1, B2, and B3. Each satellite image was segmented into 400 images of size 256×256 pixels, covering an area of approximately 59 $km^2$ each. To test the capability of our model to address more complex losses beyond scanline corruption in satellite images, we utilized a manually drawn mask dataset [Quick Draw Irregular Mask Dataset](https://github.com/karfly/qd-imd).




