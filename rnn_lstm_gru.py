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
from tqdm import tqdm

#initializ wandb project
with wandb.init(project="RNN_LSTM_GRU"):

    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Hyperparameters
    #input is batch_sizex1x28x28, so view here as 28 time points with dimension 28 each 
    #you would not use normally a RNN for images, just for practise
    input_size = 28 
    sequence_length = 28
    num_layers = 2
    hidden_size = 256
    num_classes = 10
    learning_rate = 0.005
    batch_size = 64
    num_epochs = 3

    #Create simple RNN
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(RNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.rnn = nn.RNN( #change to RNN/GRU
                input_size, #sequence length is flexible
                hidden_size,
                num_layers,
                batch_first = True, #since first axis of data is batches dim   
            )
            self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

        def forward(self, x):
            #initialize hidden state
            h0 = torch.zeros(
                self.num_layers, 
                x.size(0), #size of batch we send in
                self.hidden_size
                ).to(device)

            #forward prop
            out, _ = self.rnn(x, h0) #_ since we do not want to store hidden state explicitly
            out = out.reshape(out.shape[0], -1) #seq_length(28)*hidden_size(256) = 7168 #size of batch stays same
            out = self.fc(out)
            return out


    #Load Data
    train_dataset = datasets.MNIST(root="dataset/", train=True, transform = transforms.ToTensor(), download= True)
    #downloads dataset if it is not in root folder, transform makes NumPy array input into tensor Pytorch can work with
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform = transforms.ToTensor(), download= True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #Initialize network
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)


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
            data = data.to(device).squeeze(1) #moves datato gpu/cpu
            targets = targets.to(device=device) #moves labels

            #flattening from NN not needed
            #data = data.reshape(data.shape[0], -1)

            #forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            #backward pass
            optimizer.zero_grad() #set gradients to 0 for next optimizer step
            loss.backward()

            #gradient descent or adam step
            optimizer.step()
            wandb.log({"epoch": epoch, "loss": loss, "scores": scores})

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
                x = x.to(device=device).squeeze(1)
                y = y.to(device=device)
                x = x.reshape(x.shape[0], -1)

                scores = model(x)
                #shape of scores: 64x10, we want to know max of 2nd dimension (class labels)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
                wandb.log({"accuracy": num_correct / num_samples})

    
            #.2f just prints to decimals

        model.train()
        return num_correct / num_samples


    
    print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
    print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")

