#imports
import wandb
import torch
import torch.nn as nn #NN modules like nn.linear, nn.conv etc,
#also loss fcts
import torch.optim as optim #optimizations alg (SGD, adam,..)
import torch.nn.functional as F #fcts without parametrs,
#such as activation fct (ReLu, sigmoid, tanh), also in nn included
from torch.utils.data import DataLoader #easier data set management
import torchvision.datasets as datasets #given datasets
import torchvision.transforms as transforms #transformatations for data sets
from tqdm import tqdm  # For nice progress bar!

#initializ wandb project
with wandb.init(project="simple_CNN"):

    #Create simple CNN
    class CNN(nn.Module):
        def __init__(self, in_channels = 1, num_classes=10):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(
                in_channels= in_channels,
                out_channels=8, 
                kernel_size=(3,3), 
                stride=(1,1), 
                padding=(1,1),
            )
            #same convolution: dimensions stay the same (28x28)
            #num_out = ((num_in + 2*pad_size -kern_size)/stride_size) ; floor of this +1
            #num_out = ((28 + 2*1 -3)/1) = 27; num_out = 28
            self.pool = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2),)
            self.conv2 = nn.Conv2d(
                in_channels=8, 
                out_channels=16, 
                kernel_size=(3,3), 
                stride=(1,1), 
                padding=(1,1),
            )
            self.fc1 = nn.Linear(16*7*7, num_classes)

        def forward(self, x):
            #print(x.shape)
            x = F.relu(self.conv1(x))
            #print(x.shape)
            x = self.pool(x)
            #print(x.shape)
            x = F.relu(self.conv2(x))
            #print(x.shape)
            x = self.pool(x)
            #print(x.shape)
            x = x.reshape(x.shape[0], -1) #keep minibatch size and flatten output otherwise
            #print(x.shape)
            x = self.fc1(x)
            return x

    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Hyperparameters
    in_channels = 1
    num_classes = 10
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 3
    wandb.config = {"learning_rate": learning_rate, "epochs": num_epochs, "batch_size": batch_size}

    #Load Data
    train_dataset = datasets.MNIST(root="dataset/", train=True, transform = transforms.ToTensor(), download= True)
    #downloads dataset if it is not in root folder, transform makes NumPy array input into tensor Pytorch can work with
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform = transforms.ToTensor(), download= True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #Initialize network
    model = CNN(in_channels= in_channels, num_classes= num_classes).to(device) #no parameters needed since we set them default


    #Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #Train Network
    wandb.watch(model, criterion, log="all", log_freq=10)
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            #enumerate tells us which batch_idx we have
            #data are images, targets is correct label for each image
            #Get data to cuda if possible
            data = data.to(device) #moves datato gpu/cpu
            targets = targets.to(device=device) #moves labels

            #flattening not needed since we do that during forward pass already
            #data = data.reshape(data.shape[0], -1)

            #forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            #backward pass
            optimizer.zero_grad()
            loss.backward()

            #gradient descent or adam step
            optimizer.step()
            wandb.log({"epoch": epoch, "loss": loss, "scores": scores})

    #Check accuracy on training & test to see how good our model is
    def check_accuracy(loader, model):
        num_correct = 0
        num_samples = 0
        model.eval() #puts model in test mode (turns off dropout/batchnorms/..)

        with torch.no_grad(): #reduces memory consumption during inference time
            #by not computing any gradients 
            for x,y in loader:
                x = x.to(device=device)
                y = y.to(device=device)
                #x = x.reshape(x.shape[0], -1)
                #again reshape no longer needed since we pass 3D tensors, not flattened tensors
                scores = model(x)
                #shape of scores: 64x10, we want to know max of 2nd dimension (class labels)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            wandb.log({"accuracy": num_correct / num_samples})

        model.train()
        return num_correct / num_samples

print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")

