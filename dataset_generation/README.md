# Dataset Generation (RailSem19Cropped, FishyrailsCropped)

This directory contains the code to create the *RailSem19Cropped* and *FishyrailsCropped* datasets.

## Set-up

### Datasets 
Download the RailSem19 dataset from https://wilddash.cc/railsem19 and store it under /path/to/datasets/rs19_val.

Download PascalVOC dataset from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit and store it under /path/to/datasets/VOC2012.

Download ImageNet dataset from https://image-net.org/download.php and store it under /path/to/datasets/ImageNet.

### Virtual Environment (using Python 3.8)
```
cd railway-anomaly-detection/dataset_generation
mkdir venv
python3 -m venv venv/dataset_generation
source venv/dataset_generation/bin/activate
pip install opencv-python
pip install scipy
pip install pyyaml
pip install matplotlib
pip install scikit-image
pip install python-xml2dict
pip install h5py
```

## Dataset Creation

### Create Railsem19Croppedv1
```
# Generate region of interest crops
python3 generate_image_crops.py --max_images 999999 --mode rs19 --input_path /path/to/datasets/rs19_val --output_path /path/to/datasets/Railsem19Croppedv1

# Convert dataset to hdf5
python3 railsem19cropped2hdf5 --input_path /path/to/datasets/Railsem19Croppedv1 --output_name Railsem19Croppedv1
```

### Create FishyrailsCroppedv1
```
# Augment RailSem19 with obstacles
python3 fishyrails.py --max_images 1000 --max_obstacles 2000 --output_path /path/to/datasets/Fishyrailsv1 --input_path_rs19 /path/to/datasets/rs19_val --input_path_voc /path/to/datasets/VOC2012

# Generate region of interest crops
python3 generate_image_crops --max_images 999999 --mode fishyrails --input_path /path/to/datasets/Fishyrailsv1 --output_path /path/to/datasets/FishyrailsCroppedv1

# Convert dataset to hdf5
python3 fishyrailscropped2hdf5.py --input_path /path/to/datasets/FishyrailsCroppedv1 --output_name FishyrailsCroppedv1
```

### ImageNet
```
# Convert dataset to hdf5
python3 imagenet2hdf5.py --input_path /path/to/datasets/ImageNet --output_name ImageNet
```