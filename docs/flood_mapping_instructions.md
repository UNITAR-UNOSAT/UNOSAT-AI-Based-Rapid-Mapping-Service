# Training Instruction

Last update: May 31, 2023

## Step 1: Download and pre-process images

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

## Run training

**XNet and U-Net Training:**

Edit the configuration file and run: 

```bash
cd naive_segmentation
python model_training.py --config_file /configs/config_example.yaml
```

**Fastai U-Net Training:**

See notebook https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/Fastai%20training.ipynb

## Step 4: Train the Model

### XNet and U-Net Inference
```bash
cd naive_segmentation
python model_inference.py --config_file /configs/config_example.yaml
```

