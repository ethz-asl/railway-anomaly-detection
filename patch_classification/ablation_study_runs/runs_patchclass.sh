# Patch size 21 

# PatchClass
python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGANv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 1 --save_freq 1 --optimize_with_mask 1 --use_gan 0 --run_name patchclass_21

# PatchDiff (MSE)
python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedMSEv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 1 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_21_mse

# PatchDiff (SSIM)
python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedSSIMv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 1 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_21_ssim

# PatchDiff (GAN)
python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGANv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 1 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_21_gan

# PatchDiff (GAN+HIST)
python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGAN+HISTv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 1 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_21_gan+hist


# Patch size 13

# PatchClass
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGANv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 0 --save_freq 1 --optimize_with_mask 1 --use_gan 0 --run_name patchclass_13

# PatchDiff (MSE)
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedMSEv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 0 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_13_mse

# PatchDiff (SSIM)
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedSSIMv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 0 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_13_ssim

# PatchDiff (GAN)
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGANv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 0 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_13_gan

# PatchDiff (GAN+HIST)
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGAN+HISTv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 0 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_13_gan+hist


# Patch size 29

# PatchClass
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGANv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 2 --save_freq 1 --optimize_with_mask 1 --use_gan 0 --run_name patchclass_29

# PatchDiff (MSE)
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedMSEv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 2 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_29_mse

# PatchDiff (SSIM)
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedSSIMv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 2 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_29_ssim

# PatchDiff (GAN)
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGANv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 2 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_29_gan

# PatchDiff (GAN+HIST)
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGAN+HISTv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 2 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_29_gan+hist


# Patch size 35

# PatchClass
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGANv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 3 --save_freq 1 --optimize_with_mask 1 --use_gan 0 --run_name patchclass_35

# PatchDiff (MSE)
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedMSEv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 3 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_35_mse

# PatchDiff (SSIM)
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedSSIMv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 3 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_35_ssim

# PatchDiff (GAN)
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGANv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 3 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_35_gan

# PatchDiff (GAN+HIST)
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGAN+HISTv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 3 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_35_gan+hist


# Patch size 51

# PatchClass
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGANv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 4 --save_freq 1 --optimize_with_mask 1 --use_gan 0 --run_name patchclass_51

# PatchDiff (MSE)
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedMSEv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 4 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_51_mse

# PatchDiff (SSIM)
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedSSIMv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 4 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_51_ssim

# PatchDiff (GAN)
# python train_patchclass.py --device cpu --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGANv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 4 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_51_gan

# PatchDiff (GAN+HIST)
# python train_patchclass.py --device cpua --lr 0.1 --batch_size 32 --output_dir ./trained_models --data_path ./datasets/Railsem19CroppedGAN+HISTv1.h5 --data_path_in ./datasets/ImageNet.h5 --model patchclassmodel --stages 4 --save_freq 1 --optimize_with_mask 1 --use_gan 1 --run_name patchdiff_51_gan+hist

