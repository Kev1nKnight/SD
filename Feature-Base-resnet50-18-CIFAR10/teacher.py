import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
import torchvision
import torchvision.models as models
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.cuda.manual_seed(0)


class TeacherNet_resp(nn.Module):
  def __init__(self):
    super(TeacherNet_resp,self).__init__()
    self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # self.resnet50.conv1 = nn.Conv2d(3,64,3,1)
    # self.resnet50.maxpool = nn.Identity()
    in_features = self.resnet50.fc.in_features
    self.resnet50.fc = nn.Linear(in_features,10)

  def forward(self,x):
    output = self.resnet50(x)
    
    return output

def train_teacher_resp(model,device,train_loader,optimizer,epoch):
  model.train()
  trained_samples = 0
  
  for batch_idx,(data,target) in enumerate(train_loader):
    data,target = data.to(device),target.to(device)
    output = model(data)
    optimizer.zero_grad()
    loss = F.cross_entropy(output,target)
    loss.backward()
    optimizer.step()
    
    trained_samples += len(data)
    progress = math.ceil((batch_idx) / len(train_loader) * 50)
    print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
            (epoch, trained_samples,len(train_loader.dataset),'-' * progress + ">", progress * 2),end="")
    
def test_teacher_resp(model,device,test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data,target in test_loader:
      data,target = data.to(device),target.to(device)
      output = model(data)
      test_loss += F.cross_entropy(output,target,reduction="sum").item()
      pred = output.argmax(dim=1, keepdim=True)  
      correct += pred.eq(target.view_as(pred)).sum().item()
  test_loss /= len(test_loader.dataset)



  print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
  return test_loss, correct / len(test_loader.dataset)

  
def main_teacher_resp():
  batch_size = 64
  epochs = 1
  torch.manual_seed(0)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  transform_list = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
  ])

  train_dataset = torchvision.datasets.CIFAR10("./data/CIFAR10",train=True,download=True,transform=transform_list)
  train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,shuffle=True)
  test_dataset = torchvision.datasets.CIFAR10("./data/CIFAR10",train=False,download=True,transform=transform_list)
  test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = batch_size,shuffle=True)
  model = TeacherNet_resp().to(device)
  #adadelta ==> very very very very slow
  # optimizer = torch.optim.Adadelta(model.parameters())
  lr = 0.001
  optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)
  teacher_history = []
  for epoch in range(1,epochs+1):
    train_teacher_resp(model,device,train_loader,optimizer,epoch)
    loss,acc = test_teacher_resp(model,device,test_loader)

  torch.save(model.state_dict(), "teacher.pt")
  return model, teacher_history

