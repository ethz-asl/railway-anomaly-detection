# Patch Classification

This directory contains the code to training and evaluation of Auto-encoders, Patch Classification Networks, Patch Difference Networks, and Baseline Models. 

## Set-Up

### Copy datasets and trained models

First, you need to generate the two datasets *RailSem19Cropped.h5* and *FishyrailsCropped.h5* following the instructions in the dataset_generation directory. 
Make sure that these datasets are contained in the patch_classification/datasets directory or symlinked to it. 

```
# Copy datasets
cd railway-anomaly-detection/patch_classification
mkdir datasets
cp /path/to/datasets/* datasets # alternatively, use symlink
```

If you only want to re-train or evaluate some of the networks, we provide all the trained models that are used in this README. 
In order to use them, copy them into the patch_classification/trained_models directory.

```
# Copy trained models (if you do not want to train from scratch)
cd railway-anomaly-detection/patch_classification
mkdir trained_models
cp /path/to/trained_models/* trained_models # alternatively, use symlink

# Create evaluations directory
mkdir evaluations
```

### Virtual Environment (using Python 3.8)

```
cd railway-anomaly-detection/patch_classification 
mkdir venv
python3 -m venv venv/patch_classification
source venv/patch_classification/bin/activate
pip install opencv-python
pip install scipy
pip install torch
pip install torchvision # different version might be required for GPU
pip install wheel
pip install tensorboard
pip install h5py
pip install torchmetrics
pip install torchgeometry 
pip install scikit-learn
pip install scikit-image
pip install matplotlib
# Add to Jupyter
pip install notebook
python -m ipykernel install --user --name=patch_classification
```

## Create masks for Real-World dataset
First, create a directory with images similar to the real_world_dataset_raw directory. 
In principle, the skript can process images of any size, they are simply scaled to 224x224 and thenmrescaled back. 
However, the model works best if the images are already cropped similar to the ones the model was trained on (so 224x224).
By setting the threshold high (e.g. 0.9), obstacles are more often classified as railway track, which is what we want.
(The best obstacle detection performance is for 0.3, so rather low). Simply run
```
cd railway-anomaly-detection/patch_classification 
python real_world_masking --threshold 0.9 --data_path /path/to/dataset_raw/directory # you can use --visualize 1 to enable visualizations (stored in output directory)
```



## Evaluate on Real-World dataset

First, create a directory with your dataset similar to the real_world_dataset directory. 
Images should be in <image_name>.png format, the corresponding railway segmentation masks should be named 
<image_name>_mask.png, where 1s denote pixels that contain railway and 0 pixels that do not contain railway.

There are two possibilities to provide annotations. Either use an <image_name>_obstacle.png file, which contains a mask
with 1s at the obstacle locations and 0s everywhere else. This allows for computation of both AUROC and F1 score. 
For this option, use 
```
cd railway-anomaly-detection/patch_classification 
mkdir real_world_results # that's where results are stored  
python real_world_evaluation --config patchclass --data_path /path/to/dataset/directory --obstacle_segmentation 1 # check -h to see options for config
```

Alternatively, use annotation files <image_name>.txt (<image_name>_obstacle.png files are ignored if present), 
which should contain 5 integers separated by a space.
The first one contains a 1 if there is an obstacle present and a 0 if not. The last 4 integers indicate the 
left-most, up-most, right-most, down-most pixels of the obstacle bounding box. Only F1 score is computed, since no
obstacle mask is used. For this option, use 
```
cd railway-anomaly-detection/patch_classification 
mkdir real_world_results # that's where results are stored  
python real_world_evaluation --config patchclass --data_path /path/to/dataset/directory --obstacle_segmentation 0 # check -h to see options for config
```

## Train Auto-encoder

You can train auto-encoders with the following commands:
```
# MSE AE
python train_autoencoder.py --device cpu --lr 0.1 --momentum 0.9 --batch_size 64 --data_path ./datasets/Railsem19Croppedv1.h5 --output_dir ./trained_models --optimize_with_mask 0 --w_mse 1.0 --w_ssim 0.0 --w_gan 0.0 --w_emd 0.0 --w_mi 0.0 --optimizer adam --patience 10 --run_name ae_mse

# SSIM AE
python train_autoencoder.py --device cpu --lr 0.1 --momentum 0.9 --batch_size 64 --data_path ./datasets/Railsem19Croppedv1.h5 --output_dir ./trained_models --optimize_with_mask 0 --w_mse 0.0 --w_ssim 1.0 --w_gan 0.0 --w_emd 0.0 --w_mi 0.0 --optimizer adam --patience 10 --run_name ae_ssim

# GAN AE
python train_autoencoder.py --device cpu --lr 0.0001 --lr_d 0.0001 --momentum 0.5 --batch_size 64 --data_path ./datasets/Railsem19Croppedv1.h5 --output_dir ./trained_models --optimize_with_mask 0 --w_mse 0.0 --w_ssim 0.0 --w_gan 1.0 --w_emd 0.0 --w_mi 0.0 --optimizer adam --patience 200 --run_name ae_gan

# GAN+HIST AE
python train_autoencoder.py --device cpu --lr 0.0001 --lr_d 0.0001 --momentum 0.5 --batch_size 16 --data_path ./datasets/Railsem19Croppedv1.h5 --output_dir ./trained_models --optimize_with_mask 0 --w_mse 0.0 --w_ssim 0.0 --w_gan 1.0 --w_emd 1.0 --w_mi 1.0 --optimizer adam --patience 200 --run_name ae_gan+hist
```
For training on GPU, change the device argument to cuda. The outputs are stored in trained_models/RUNNAME_DATE_TIME.

You can monitor training progress on tensorboard via 
```
tensorboard --logdir tensorboard_runs
```
Training can be resumed from checkpoints via the resume argument. 

After training for 200 epochs, copy the model to the trained_models directory.
```
cp ./trained_models/ae_mse_DATE_TIME/model_199.pth ./trained_models/ae_mse_model_199.pth
cp ./trained_models/ae_ssim_DATE_TIME/model_199.pth ./trained_models/ae_ssim_model_199.pth
cp ./trained_models/ae_gan_DATE_TIME/model_199.pth ./trained_models/ae_gan_model_199.pth
cp ./trained_models/ae_gan+hist_DATE_TIME/model_199.pth ./trained_models/ae_gan+hist_model_199.pth
```

### Create auto-encoder datasets
In order to speed up training of PatchDiff models, it is required to add the auto-encoder generated images to the RailSem19Cropped training dataset. 
```
# Railsem19CroppedMSEv1
python create_ae_dataset.py --input_path ./datasets/Railsem19Croppedv1.h5 --checkpoint ./trained_models/ae_mse_model_199.pth --output_path ./datasets/Railsem19CroppedMSEv1.h5

# Railsem19CroppedSSIMv1
python create_ae_dataset.py --input_path ./datasets/Railsem19Croppedv1.h5 --checkpoint ./trained_models/ae_ssim_model_199.pth --output_path ./datasets/Railsem19CroppedSSIMv1.h5

# Railsem19CroppedGANv1
python create_ae_dataset.py --input_path ./datasets/Railsem19Croppedv1.h5 --checkpoint ./trained_models/ae_gan_model_199.pth --output_path ./datasets/Railsem19CroppedGANv1.h5

# Railsem19CroppedGAN+HISTv1
python create_ae_dataset.py --input_path ./datasets/Railsem19Croppedv1.h5 --checkpoint ./trained_models/ae_gan+hist_model_199.pth --output_path ./datasets/Railsem19CroppedGAN+HISTv1.h5
```

## Train PatchClass / PatchDiff

These are commands for training our PatchClass / PatchDiff models for a patch size of K_p = 21. For other patch sizes, have a look at ablation_study_runs/runs_patchclass.sh 
```
# PatchClass21
python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGANv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 1 --save_freq 1 --optimize_with_mask 1 --use_gan 0 --run_name patchclass_21

# PatchDiff21 (MSE)
python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedMSEv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 1 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_21_mse

# PatchDiff21 (SSIM)
python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedSSIMv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 1 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_21_ssim

# PatchDiff21 (GAN)
python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGANv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 1 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_21_gan

# PatchDiff21 (GAN+HIST)
python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGAN+HISTv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 1 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_21_gan+hist
```
For training on GPU, change the device argument to cuda. The outputs are stored in trained_models/RUNNAME_DATE_TIME.

You can monitor training progress on tensorboard via 
```
tensorboard --logdir tensorboard_runs
```

After training for 30 epochs, copy the best model to the trained_models directory.
```
cp ./trained_models/patchclass_21_DATE_TIME/model_20.pth ./trained_models/patchclass_21_model_20.pth
cp ./trained_models/patchdiff_21_mse_DATE_TIME/model_25.pth ./trained_models/patchdiff_21_mse_model_25.pth
cp ./trained_models/patchdiff_21_ssim_DATE_TIME/model_20.pth ./trained_models/patchdiff_21_ssim_model_20.pth
cp ./trained_models/patchdiff_21_gan_DATE_TIME/model_30.pth ./trained_models/patchdiff_21_gan_model_30.pth
cp ./trained_models/patchdiff_21_gan+hist_DATE_TIME/model_20.pth ./trained_models/patchdiff_21_gan+hist_model_20.pth
```

## Evaluate PatchClass / PatchDiff

The final models can be evaluated with the following commands. ROC AUC and F1 score are reported in evaluations/RUNNAME/output.txt, visualizations for theta_visualize are provided in evaluations/RUNNAME/ . K_d can be changed with the respective argument. For different patch sizes K_p, have a look at evaluations_patchclass.sh.
```
# PatchClass21 
python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch3 --stages 1 --k_d 11 --theta_visualize 0.95 --checkpoint ./trained_models/patchclass_21_model_20.pth --ae_model none --output_path ./evaluations/patchclass_21

# PatchDiff21 (MSE)
python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 1 --k_d 7 --theta_visualize 0.9 --checkpoint ./trained_models/patchdiff_21_mse_model_25.pth --ae_checkpoint ./trained_models/ae_mse_model_199.pth --output_path ./evaluations/patchdiff_21_mse

# PatchDiff21 (SSIM)
python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 1 --k_d 7 --theta_visualize 0.95 --checkpoint ./trained_models/patchdiff_21_ssim_model_20.pth --ae_checkpoint ./trained_models/ae_ssim_model_199.pth --output_path ./evaluations/patchdiff_21_ssim

# PatchDiff21 (GAN)
python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 1 --k_d 11 --theta_visualize 0.95 --checkpoint ./trained_models/patchdiff_21_gan_model_30.pth --ae_checkpoint ./trained_models/ae_gan_model_199.pth --output_path ./evaluations/patchdiff_21_gan

# PatchDiff21 (GAN+HIST)
python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 1 --k_d 7 --theta_visualize 0.95 --checkpoint ./trained_models/patchdiff_21_gan+hist_model_20.pth --ae_checkpoint ./trained_models/ae_gan+hist_model_199.pth --output_path ./evaluations/patchdiff_21_gan+hist
```

## Train and Evaluate Baselines

### DeeplabV3 Standard Semantic Segmentation
First, train the standard semantic segmentation model:
```
python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGANv1.h5 --model deeplabv3_resnet50 --stages 1 --save_freq 1 --optimize_with_mask 0 --use_gan 0 --imagenet_ratio 0.0 --run_name deeplabv3
```
Then, copy the model to trained_models and evaluate it.
```
cp ./trained_models/deeplabv3_DATE_TIME/model_5.pth ./trained_models/deeplabv3_model_5.pth
python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model deeplabv3_resnet50 --k_d 51 --theta_visualize 0.3 --checkpoint ./trained_models/deeplabv3_model_5.pth --ae_model none --output_path ./evaluations/deeplabv3
```
### RMSE and SSIM Auto-encoders
Since we already trained the auto-encoders, we can directly evaluate them:
```
# RMSE AE
python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model mse --k_d 7 --theta_visualize 0.2 --ae_checkpoint ./trained_models/ae_mse_model_199.pth --output_path ./evaluations/ae_rmse

# SSIM AE
python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model ssim --k_d 21 --theta_visualize 0.65 --ae_checkpoint ./trained_models/ae_ssim_model_199.pth --output_path ./evaluations/ae_ssim
```

### Student Teacher 
Here, we focus on the student teacher method with patch size 33. for other patch sizes, have a look at runs_teacher.sh, runs_students.sh, and evaluations_students.sh in directory ablation_study_runs.

First, train the teacher model.
```
python train_teacher.py --device cpu --lr 0.0002 --batch_size 64 --device cuda --output_dir ./trained_models --data_path ./datasets/ImageNet.h5 --stages 2 --patch_size 32 --save_freq 1 --run_name teacher_33
```
After training for 50 epochs (= 50,000 iterations), copy the model to trained_models and compute the teacher's mean and standard deviation over the railway training dataset.
```
cp ./trained_models/teacher_33_DATE_TIME/model_49.pth ./trained_models/teacher_33_model_49.pth
python compute_mean_teacher.py --model patchsegmodellight --device cuda --stages 2 --teacher_checkpoint ./trained_models/teacher_33_model_49.pth
```
Now, train the three students:
```
python train_student.py --device cpu --lr 0.0001 --batch_size 1 --teacher_checkpoint ./trained_models/teacher_33_model_49.pth --output_dir ./trained_models --stages 2 --save_freq 1 --seed 1 --run_name t33_student_1
python train_student.py --device cpu --lr 0.0001 --batch_size 1 --teacher_checkpoint ./trained_models/teacher_33_model_49.pth --output_dir ./trained_models --stages 2 --save_freq 1 --seed 2 --run_name t33_student_2
python train_student.py --device cpu --lr 0.0001 --batch_size 1 --teacher_checkpoint ./trained_models/teacher_33_model_49.pth --output_dir ./trained_models --stages 2 --save_freq 1 --seed 3 --run_name t33_student_3
```
After training until convergence, copy the models to trained_models and evaluate the ensemble of students
```
cp ./trained_models/t33_student_1_DATE_TIME/model_40.pth ./trained_models/t33_student_1_model_40.pth
cp ./trained_models/t33_student_2_DATE_TIME/model_40.pth ./trained_models/t33_student_2_model_40.pth
cp ./trained_models/t33_student_3_DATE_TIME/model_40.pth ./trained_models/t33_student_3_model_40.pth

python evaluate_students_fishyrails.py --device cpu --data_path_test ./datasets/FishyrailsCroppedv1.h5 --data_path_val ./datasets/Railsem19Croppedv1.h5 --teacher_checkpoint ./trained_models/teacher_33_model_49.pth --student1_checkpoint ./trained_models/t33_student_1_model_40.pth --student2_checkpoint ./trained_models/t33_student_2_model_40.pth --student3_checkpoint ./trained_models/t33_student_3_model_40.pth --stages 2 --k_d 35 --theta_visualize 0.2 --output_path ./evaluations/students_33 
```


## Run on Euler 

### One-Time Set-Up
```
# Load correct modules for first time (need to switch to new software stack)
set_software_stack.sh new
module load gcc/8.2.0 python_gpu/3.8.5
# create venv
cd /path/to/railway-anomaly-detection
python3 -m venv --system-site-packages euler-venv 
source euler-venv/bin/activate
pip install torchgeometry
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu111_pyt191/download.html
```

###  Every Time 
```
ssh USER@euler.ethz.ch
module load gcc/8.2.0 python_gpu/3.8.5
source euler-venv/bin/activate
bsub -W 240 -Is -R "rusage[mem=20000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=20000]" bash # for interactive experimentation
bsub -W 1440 -o output.txt -R "rusage[mem=20000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=20000]" COMMAND # for training
```