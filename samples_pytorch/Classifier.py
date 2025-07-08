import torch.nn as nn 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Classifier(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
        nn.Linear(D_in, H),
        nn.Sigmoid() ,
        
        nn.Linear(H, D_out)
        )
        
    def forward(self, x):
        return self.network(x)


def accuracy_fn(y, yhat):
    preds = torch.argmax(yhat, dim=1)
    return (preds == y).float().mean().item()

def train(model, optimizer, criterion, trainloader, testloader, epochs):
    results = {'train_loss': [], 'validation_loss': [], 'accuracy':[]}
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for xbatch, ybatch in trainloader:
            xbatch = xbatch.view(xbatch.size(0), -1)
            optimizer.zero_grad() 
            yhat = model(xbatch)
            loss = criterion(yhat, ybatch)
            running_loss += loss.item() 
                
            loss.backward()
            optimizer.step()
        train_cost = running_loss / len(trainloader)
        results['train_loss'].append(train_cost)    
            
            
        # validation 
        model.eval() 
        val_loss = 0
        accuracy_epoch = 0
        for xval, yval in testloader:
            xval = xval.view(xval.size(0), -1)
            yhat = model(xval)
            loss = criterion(yhat, yval)
            val_loss += loss.item()
            acc = accuracy_fn(yval, yhat)
            accuracy_epoch += acc
        
        results['accuracy'].append(accuracy_epoch / len(testloader))
        results ['validation_loss'].append(val_loss / len(testloader))
    
    return results


epochs=30
weight_decay = 0.01
learning_rate = 0.1
D_in = 28 * 28 
H = 5
D_out = 10 
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
])

torch.manual_seed(42)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)





    
model = Classifier(D_in, H, D_out)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
results = train(model, optimizer, criterion, trainloader, testloader, epochs)
