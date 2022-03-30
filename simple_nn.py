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

#initializ wandb project
with wandb.init(project="simple_fcnn"):

    #create fully connected network
    class NN(nn.Module):
        def __init__(self, input_size, num_classes):
            super(NN, self).__init__() #run initialization of nn.Module
            self.fc1 = nn.Linear(input_size, 50) #50 dim. hidden layer (very small)
            self.fc2 = nn.Linear(50, num_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = NN(784, 10)
    x = torch.randn(64, 784)

    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Hyperparameters
    input_size = 784
    num_classes = 10
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 1

    #Load Data
    train_dataset = datasets.MNIST(root="dataset/", train=True, transform = transforms.ToTensor(), download= True)
    #downloads dataset if it is not in root folder, transform makes NumPy array input into tensor Pytorch can work with
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform = transforms.ToTensor(), download= True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #Initialize network
    model = NN(input_size=input_size, num_classes=num_classes).to(device)


    #Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #Train Network
    wandb.watch(model, criterion, log="all", log_freq=10)
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            #enumerate tells us which batch_idx we have
            #data are images, targets is correct label for each image
            #Get data to cuda if possible
            data = data.to(device) #moves datato gpu/cpu
            targets = targets.to(device=device) #moves labels

            #flatten 28x28 image to correct shape
            data = data.reshape(data.shape[0], -1)

            #forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            #backward pass
            optimizer.zero_grad()
            loss.backward()

            #gradient descent or adam step
            optimizer.step()
            wandb.log({"epoch": epoch, "loss": loss}, step=10)

    #Check accuracy on training & test to see how good our model is
    def check_accuracy(loader, model):
        if loader.dataset.train:
            print("Checking accuracy on training data")
        else:
            print("Checking accuracy on test data")
        num_correct = 0
        num_samples = 0
        model.eval() #puts model in test mode (turns off dropout/batchnorms/..)

        with torch.no_grad(): #reduces memory consumption during inference time
            #by not computing any gradients 
            for x,y in loader:
                x = x.to(device=device)
                y = y.to(device=device)
                x = x.reshape(x.shape[0], -1)

                scores = model(x)
                #shape of scores: 64x10, we want to know max of 2nd dimension (class labels)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            wandb.log({"test_accuracy": num_correct / num_samples})
            print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
            #.2f just prints to decimals

        model.train()

    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)

