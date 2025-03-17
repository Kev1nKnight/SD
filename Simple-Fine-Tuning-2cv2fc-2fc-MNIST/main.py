from teacher import * 
from student_kd import * 
from matplotlib import pyplot as plt
import torch.nn.functional as F 
import numpy as np
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
teacher_model, teacher_history = teacher_main()
student_kd_model, student_kd_history = student_kd_main(teacher_model)
student_simple_model, student_simple_history = student_main(teacher_model)

epochs = 10
x = list(range(1, epochs+1))

plt.subplot(2, 1, 1)
plt.plot(x, [teacher_history[i][1] for i in range(epochs)], label='teacher')
plt.plot(x, [student_kd_history[i][1] for i in range(epochs)], label='student with KD')
plt.plot(x, [student_simple_history[i][1] for i in range(epochs)], label='student without KD')

plt.title('Test accuracy')
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(x, [teacher_history[i][0] for i in range(epochs)], label='teacher')
plt.plot(x, [student_kd_history[i][0] for i in range(epochs)], label='student with KD')
plt.plot(x, [student_simple_history[i][0] for i in range(epochs)], label='student without KD')

plt.title('Test loss')
plt.legend()
