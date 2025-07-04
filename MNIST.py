import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 100
input_size = 784 # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 2
learning_rate = 0.001


train_data = torchvision.datasets.MNIST(
    root="./data",
    train = True,
    transform = transforms.ToTensor(),
    download  = True
)
test_data = torchvision.datasets.MNIST(
    root="./data",
    train = False,
    transform = transforms.ToTensor(),
)

train_loader = DataLoader(
    dataset = train_data,
    batch_size= batch_size,
    shuffle = True
)

test_loader = DataLoader(
    dataset = test_data,
    batch_size= batch_size,
    shuffle = False
)

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size,num_classes)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

model = NeuralNet(input_size,hidden_size,num_classes).to(device)

evaluator = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i , (images,labels) in enumerate(train_loader):
        # images: torch.Tensor
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)

        output = model(images)
        loss = evaluator(output,labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

with torch.no_grad():
    n_correct = 0
    n_samples = len(test_loader.dataset)

    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        outputs = model(images)

        _,predicted = torch.max(outputs,1)
        n_correct += (predicted == labels).sum().item()

    acc = n_correct / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {100*acc} %')



