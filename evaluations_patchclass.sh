# Patch size 21

# No AE
python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch3 --stages 1 --k_d 11 --theta_visualize 0.95 --checkpoint ./trained_models/patchclass_21_model_20.pth --ae_model none --output_path ./evaluations/patchclass_21

# MSE
python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 1 --k_d 7 --theta_visualize 0.9 --checkpoint ./trained_models/patchdiff_21_mse_model_25.pth --ae_checkpoint ./trained_models/ae_mse_model_199.pth --output_path ./evaluations/patchdiff_21_mse

# SSIM
python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 1 --k_d 7 --theta_visualize 0.95 --checkpoint ./trained_models/patchdiff_21_ssim_model_20.pth --ae_checkpoint ./trained_models/ae_ssim_model_199.pth --output_path ./evaluations/patchdiff_21_ssim

# GAN
python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 1 --k_d 11 --theta_visualize 0.95 --checkpoint ./trained_models/patchdiff_21_gan_model_30.pth --ae_checkpoint ./trained_models/ae_gan_model_199.pth --output_path ./evaluations/patchdiff_21_gan

# GAN+HIST
python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 1 --k_d 7 --theta_visualize 0.95 --checkpoint ./trained_models/patchdiff_21_gan+hist_model_20.pth --ae_checkpoint ./trained_models/ae_gan+hist_model_199.pth --output_path ./evaluations/patchdiff_21_gan+hist


# Patch size 13

# No AE
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch3 --stages 0 --k_d 7 --theta_visualize 0.95 --checkpoint ./trained_models/patchclass_13_model_30.pth --ae_model none --output_path ./evaluations/patchclass_13

# MSE
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 0 --k_d 7 --theta_visualize 0.95 --checkpoint ./trained_models/patchdiff_13_mse_model_25.pth --ae_checkpoint ./trained_models/ae_mse_model_199.pth --output_path ./evaluations/patchdiff_13_mse

# SSIM
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 0 --k_d 7 --theta_visualize 0.95 --checkpoint ./trained_models/patchdiff_13_ssim_model_25.pth --ae_checkpoint ./trained_models/ae_ssim_model_199.pth --output_path ./evaluations/patchdiff_13_ssim

# GAN
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 0 --k_d 7 --theta_visualize 0.95 --checkpoint ./trained_models/patchdiff_13_gan_model_30.pth --ae_checkpoint ./trained_models/ae_gan_model_199.pth --output_path ./evaluations/patchdiff_13_gan

# GAN+HIST
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 0 --k_d 7 --theta_visualize 0.95 --checkpoint ./trained_models/patchdiff_13_gan+hist_model_30.pth --ae_checkpoint ./trained_models/ae_gan+hist_model_199.pth --output_path ./evaluations/patchdiff_13_gan+hist


# Patch size 29

# No AE
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch3 --stages 2 --k_d 11 --theta_visualize 0.95 --checkpoint ./trained_models/patchclass_29_model_30.pth --ae_model none --output_path ./evaluations/patchclass_29

# MSE
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 2 --k_d 21 --theta_visualize 0.35 --checkpoint ./trained_models/patchdiff_29_mse_model_25.pth --ae_checkpoint ./trained_models/ae_mse_model_199.pth --output_path ./evaluations/patchdiff_29_mse

# SSIM
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 2 --k_d 35 --theta_visualize 0.35 --checkpoint ./trained_models/patchdiff_29_ssim_model_20.pth --ae_checkpoint ./trained_models/ae_ssim_model_199.pth --output_path ./evaluations/patchdiff_29_ssim

# GAN
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 2 --k_d 13 --theta_visualize 0.9 --checkpoint ./trained_models/patchdiff_29_gan_model_30.pth --ae_checkpoint ./trained_models/ae_gan_model_199.pth --output_path ./evaluations/patchdiff_29_gan

# GAN+HIST
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 2 --k_d 11 --theta_visualize 0.9 --checkpoint ./trained_models/patchdiff_29_gan+hist_model_30.pth --ae_checkpoint ./trained_models/ae_gan+hist_model_199.pth --output_path ./evaluations/patchdiff_29_gan+hist


# Patch size 35

# No AE
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch3 --stages 3 --k_d 11 --theta_visualize 0.95 --checkpoint ./trained_models/patchclass_35_model_30.pth --ae_model none --output_path ./evaluations/patchclass_35

# MSE
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 3 --k_d 13 --theta_visualize 0.35 --checkpoint ./trained_models/patchdiff_35_mse_model_25.pth --ae_checkpoint ./trained_models/ae_mse_model_199.pth --output_path ./evaluations/patchdiff_35_mse

# SSIM
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 3 --k_d 51 --theta_visualize 0.15 --checkpoint ./trained_models/patchdiff_35_ssim_model_20.pth --ae_checkpoint ./trained_models/ae_ssim_model_199.pth --output_path ./evaluations/patchdiff_35_ssim

# GAN
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 3 --k_d 13 --theta_visualize 0.9 --checkpoint ./trained_models/patchdiff_35_gan_model_30.pth --ae_checkpoint ./trained_models/ae_gan_model_199.pth --output_path ./evaluations/patchdiff_35_gan

# GAN+HIST
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 3 --k_d 29 --theta_visualize 0.55 --checkpoint ./trained_models/patchdiff_35_gan+hist_model_20.pth --ae_checkpoint ./trained_models/ae_gan+hist_model_199.pth --output_path ./evaluations/patchdiff_35_gan+hist


# Patch size 51

# No AE
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch3 --stages 4 --k_d 11 --theta_visualize 0.95 --checkpoint ./trained_models/patchclass_51_model_30.pth --ae_model none --output_path ./evaluations/patchclass_51

# MSE
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 4 --k_d 51 --theta_visualize 0.05 --checkpoint ./trained_models/patchdiff_51_mse_model_25.pth --ae_checkpoint ./trained_models/ae_mse_model_199.pth --output_path ./evaluations/patchdiff_51_mse

# SSIM
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 4 --k_d 51 --theta_visualize 0.05 --checkpoint ./trained_models/patchdiff_51_ssim_model_20.pth --ae_checkpoint ./trained_models/ae_ssim_model_199.pth --output_path ./evaluations/patchdiff_51_ssim

# GAN
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 4 --k_d 35 --theta_visualize 0.45 --checkpoint ./trained_models/patchdiff_51_gan_model_30.pth --ae_checkpoint ./trained_models/ae_gan_model_199.pth --output_path ./evaluations/patchdiff_51_gan

# GAN+HIST
# python evaluate_patchclass_fishyrails.py --device cpu --data_path ./datasets/FishyrailsCroppedv1.h5 --model patchclassmodel_patch6 --stages 4 --k_d 35 --theta_visualize 0.3 --checkpoint ./trained_models/patchdiff_51_gan+hist_model_20.pth --ae_checkpoint ./trained_models/ae_gan+hist_model_199.pth --output_path ./evaluations/patchdiff_51_gan+hist
