import torch
import torchvision.transforms as transforms
from torchattacks import PGD
import torchvision
from ResNet import *

# TODO
checkpoint_path = './resnet18_cifar10_new.pt' # add your model path
data_path = './' # data path

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')

transform = transforms.Compose([
    transforms.ToTensor(),
])

batch_size = 128

testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size, shuffle=False)

model = ResNet18() 
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

correct, adv_correct = 0, 0
total = 0

attack = PGD(model, eps=0.01, steps=10)
for data in testloader:
    images, labels = data[0].to(device), data[1].to(device)
    adv_images = attack(images, labels)
    outputs = model(images)
    adv_outputs = model(adv_images)
    _, predicted = torch.max(outputs.data, 1) # Probability, index
    _, adv_predicted = torch.max(adv_outputs.data, 1) # Probability, index
    total += labels.size(0) # Number of labels
    correct += (predicted == labels).sum().item() # Number of correct predictions
    adv_correct += (adv_predicted == labels).sum().item() # Number of correct predictions

print(correct/total, adv_correct/total) # Accuracy