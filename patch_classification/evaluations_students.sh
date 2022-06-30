# Students 33

python evaluate_students_fishyrails.py --device cpu --data_path_test ./datasets/FishyrailsCroppedv1.h5 --data_path_val ./datasets/Railsem19Croppedv1.h5 --teacher_checkpoint ./trained_models/teacher_33_model_49.pth --student1_checkpoint ./trained_models/t33_student_1_model_40.pth --student2_checkpoint ./trained_models/t33_student_2_model_40.pth --student3_checkpoint ./trained_models/t33_student_3_model_40.pth --stages 2 --k_d 35 --theta_visualize 0.2 --output_path ./evaluations/students_33 


# Students 17

# python evaluate_students_fishyrails.py --device cpu --data_path_test ./datasets/FishyrailsCroppedv1.h5 --data_path_val ./datasets/Railsem19Croppedv1.h5 --teacher_checkpoint ./trained_models/teacher_17_model_49.pth --student1_checkpoint ./trained_models/t17_student_1_model_28.pth --student2_checkpoint ./trained_models/t17_student_2_model_28.pth --student3_checkpoint ./trained_models/t17_student_3_model_28.pth --stages 1 --k_d 7 --theta_visualize 0.25 --output_path ./evaluations/students_17


# Students 65

# python evaluate_students_fishyrails.py --device cpu --data_path_test ./datasets/FishyrailsCroppedv1.h5 --data_path_val ./datasets/Railsem19Croppedv1.h5 --teacher_checkpoint ./trained_models/teacher_65_model_30.pth --student1_checkpoint ./trained_models/t65_student_1_model_30.pth --student2_checkpoint ./trained_models/t65_student_2_model_30.pth --student3_checkpoint ./trained_models/t65_student_3_model_28.pth --stages 3 --k_d 21 --theta_visualize 0.15 --output_path ./evaluations/students_65
