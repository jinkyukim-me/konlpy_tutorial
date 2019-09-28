import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


batch_size = 100
total_epoch = 50
learning_rate = 0.01
device = torch.cuda_is_available()
criterion = nn.CrossEntropyLoss()



print(use_cuda)

train_dataset = dsets.CIFAR10(root='./data',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)
test_dataset = dsets.CIFAR10(root='./data',
                              train=True,
                              transform=transforms.ToTensor(),)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

class SimpleCNN(nn.modules):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        #convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        #fully-connected layer
        self.fc1 = nn.Linear(256*2*2, 1000)
        self.act1 = nn.ReLU()
        self.output = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.pool1(self.act1(Self.conv1(x)))
        x = self.pool2(self.act2(Self.conv1(x)))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.pool3(x)

        x = x.view(-1, 256*2*2)

        x = self.act5(self.fc1(x))
        out = self.output(x)
        return out

optimizer = optim.SGD(SimpleCNN.parameters(), momentum=0.9, learning_rate)

model = SimpleCNN.to(device)

criterion = criterion.to(device)

def train(model, train_loader):
    model.train()

    losses = []
    for i, (image, label) in enumerate(train_loader):
        if use_cuda:
            image = image.cuda()
            label = label.cuda()

        pred_label = model(image)
        loss = criterion(pred_label, label)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = sum(losses) / len(losses)
    return avg_loss

def eval(model, test_loader):
    model.eval()
    device = next(model.parameters()).device.index

    total_cnt = 0
    correct_cnt = 0

    for i , (image, label) in enumerate(test_loader):
        if use_cuda:
            image = image.cuda()
            label = label.cuda()

            out = model(image)
            _, pred_label = torch.max(out.data, 1)
            total_cnt += image.data.size()[0]
            correct_cnt += (pred_label == label.data).sum().item()
        return correct_cnt / total_cnt

sgd_train_loss_lst = []
sgd_test_accuracy_lst = []
for epoch in range(total_epoch):
    train_loss = train(cnn_model, train_loader)
    sgd_train_loss_lst.append(train_loss)
    sgd_test_accuracy = eval(cnn_model, test_loader)
    sgd_test_accuracy_lst.append(test_accuracy)

    print(test_accuracy)


