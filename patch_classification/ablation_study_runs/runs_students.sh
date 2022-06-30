# Student runs

# Patch size 33

python train_student.py --device cpu --lr 0.0001 --batch_size 1 --teacher_checkpoint ./trained_models/teacher_33_model_49.pth --output_dir ./trained_models --stages 2 --save_freq 1 --seed 1 --run_name t33_student_1

python train_student.py --device cpu --lr 0.0001 --batch_size 1 --teacher_checkpoint ./trained_models/teacher_33_model_49.pth --output_dir ./trained_models --stages 2 --save_freq 1 --seed 2 --run_name t33_student_2

python train_student.py --device cpu --lr 0.0001 --batch_size 1 --teacher_checkpoint ./trained_models/teacher_33_model_49.pth --output_dir ./trained_models --stages 2 --save_freq 1 --seed 3 --run_name t33_student_3


# Patch size 17

# python train_student.py --device cpu --lr 0.0001 --batch_size 1 --teacher_checkpoint ./trained_models/teacher_17_model_49.pth --output_dir ./trained_models --stages 1 --save_freq 1 --seed 1 --run_name t17_student_1

# python train_student.py --device cpu --lr 0.0001 --batch_size 1 --teacher_checkpoint ./trained_models/teacher_17_model_49.pth --output_dir ./trained_models --stages 1 --save_freq 1 --seed 2 --run_name t17_student_2

# python train_student.py --device cpu --lr 0.0001 --batch_size 1 --teacher_checkpoint ./trained_models/teacher_17_model_49.pth --output_dir ./trained_models --stages 1 --save_freq 1 --seed 3 --run_name t17_student_3


# Patch size 65

# python train_student.py --device cpu --lr 0.0001 --batch_size 1 --teacher_checkpoint ./trained_models/teacher_65_model_49.pth --output_dir ./trained_models --stages 1 --save_freq 1 --seed 1 --run_name t65_student_1

# python train_student.py --device cpu --lr 0.0001 --batch_size 1 --teacher_checkpoint ./trained_models/teacher_65_model_49.pth --output_dir ./trained_models --stages 1 --save_freq 1 --seed 2 --run_name t65_student_2

# python train_student.py --device cpu --lr 0.0001 --batch_size 1 --teacher_checkpoint ./trained_models/teacher_65_model_49.pth --output_dir ./trained_models --stages 1 --save_freq 1 --seed 3 --run_name t65_student_3

