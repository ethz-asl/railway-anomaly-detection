# Teacher runs

# Patch size 33

python train_teacher.py --device cpu --lr 0.0002 --batch_size 64 --device cuda --output_dir ./trained_models --data_path ./datasets/ImageNet.h5 --stages 2 --patch_size 32 --save_freq 1 --run_name teacher_33

python compute_mean_teacher.py --model patchsegmodellight --device cuda --stages 2 --teacher_checkpoint ./trained_models/teacher_33_model_49.pth


# Patch size 17

# python train_teacher.py --device cpu --lr 0.0002 --batch_size 64 --device cuda --output_dir ./trained_models --data_path ./datasets/ImageNet.h5 --stages 1 --patch_size 16 --save_freq 1 --run_name teacher_17

# python compute_mean_teacher.py --model patchsegmodellight --device cuda --stages 1 --teacher_checkpoint ./trained_models/teacher_17_model_49.pth


# Patch size 65

# python train_teacher.py --device cpu --lr 0.0002 --batch_size 64 --device cuda --output_dir ./trained_models --data_path ./datasets/ImageNet.h5 --stages 3 --patch_size 64 --save_freq 1 --run_name teacher_65

# python compute_mean_teacher.py --model patchsegmodellight --device cuda --stages 3 --teacher_checkpoint ./trained_models/teacher_65_model_49.pth


