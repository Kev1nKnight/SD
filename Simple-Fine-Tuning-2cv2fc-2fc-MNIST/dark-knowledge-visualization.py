from teacher import * 
from matplotlib import pyplot as plt
import torch.nn.functional as F 
import numpy as np

torch.cuda.empty_cache()
teacher_model, teacher_history = teacher_main()

test_loader_bs1 = torch.utils.data.DataLoader(
    datasets.MNIST('../data/MNIST', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=1, shuffle=True)

teacher_model.eval()
with torch.no_grad():
    data, target = next(iter(test_loader_bs1))
    data, target = data.to('cuda'), target.to('cuda')
    output = teacher_model(data)

test_x = data.cpu().numpy()
y_out = output.cpu()

y_out = y_out[0, ::]
print('Output (NO softmax):', y_out)



plt.subplot(3, 1, 1)
plt.imshow(test_x[0, 0, ::])

plt.subplot(3, 1, 2)
plt.bar(list(range(10)), F.softmax(y_out/1), width=0.3)

plt.subplot(3, 1, 3)
plt.bar(list(range(10)), F.softmax(y_out/10), width=0.3)

plt.show()