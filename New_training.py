import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from ResNet import *
from torchattacks import PGD
import wandb
wandb.init(project='CIFAR10 adversarial Example')
# 실행 이름 설정
wandb.run.name = 'adversarial_learning-SGD(lr=0.1)_early40'
wandb.run.save()

# TODO
data_path = './' # data path
PATH = './' # checkpoint path

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
batch_size=128
lr = 0.1
print(f'{device} is available')

transform = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    # transforms.RandomHorizontalFlip(p=0.1),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='data_path', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='data_path', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size, shuffle=False)

model = ResNet18()
model.to(device)

critertion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
# optimizer = optim.AdamW(model.parameters(), lr = 0.05, eps = 1e-08, weight_decay = 5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

num_epochs = 400
loss_ = []
n = len(trainloader)

args = {
    "learning_rate": lr,
    "epochs": num_epochs,
    "batch_size": batch_size
}
wandb.config.update(args)
 
model.train()

# 조기 종료 변수 초기화
early_stopping_epochs = 40
best_loss = float('inf')
early_stop_counter = 0

# 학습 루프
for epoch in range(num_epochs):
    running_loss = 0.0
    attack = PGD(model, eps=0.01, steps=10)
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        adv_images = attack(images, labels)
        # 원본 이미지와 적대적 이미지를 함께 사용하여 훈련
        outputs = model(adv_images)
        
        loss = critertion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
            
    loss_.append(running_loss/n)
    print(f'{epoch+1} loss: {running_loss/n}')
    wandb.log({"Training loss": running_loss / n})
    scheduler.step()

# prediction part
    correct = 0
    total = 0
    m = len(testloader)
    with torch.no_grad(): 
        model.eval()
        valid_loss = 0.0
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) 
            total += labels.size(0) 
            correct += (predicted == labels).sum().item()
            loss = critertion(outputs, labels)
            valid_loss += loss.item()
    print(f'accuracy of 10000 test images: {100*correct/total}%')
    wandb.log({"valid loss": valid_loss / m})
    wandb.log({"accuracy of 10000 test images": 100*correct/total})

    # 검증 데이터셋의 손실이 이전보다 증가하는 경우
    if valid_loss > best_loss:
        early_stop_counter += 1
    else:
        best_loss = valid_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), PATH + 'resnet18_cifar10_new.pt')
    # 조기 종료 조건 확인
    if early_stop_counter >= early_stopping_epochs:
        print("Early Stopping!")
        break