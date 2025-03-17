from teacher import *
from student import *

torch.cuda.empty_cache()
teacher_resp_model,teacher_resp_history = main_teacher_resp()
student_model,student_history = main_student(teacher_resp_model=teacher_resp_model) 
