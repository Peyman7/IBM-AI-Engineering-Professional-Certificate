import torch
import torch.nn as nn 


class NeuralNetwork(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(NeuralNetwork, self).__init__()
        
        self.network = nn.Sequential(
        nn.Linear(D_in, H1),
        nn.Sigmoid(),
        
        nn.Linear(H1, H2),
        nn.Sigmoid(),
        
        nn.Linear(H2, D_out),
        nn.Sigmoid()
        )
        
    def forward(self, x): 
        return self.network(x)



def crietrion(yhat, y):
    out = -1 * torch.mean(y * torch.log(yhat) + (1-y)* torch.log(1-yhat))
    return out

def train(X, y, model, optimizer, criterion, epochs):
    losses = []
    model.train()
    for epoch in range(epochs):
        total = 0
        for x, ytrue in zip(X, y):
            x = x.unsqueeze(0)                 # shape: (1, D_in)
            ytrue = ytrue.unsqueeze(0).unsqueeze(1)  # shape: (1, 1)
            optimizer.zero_grad()
            yhat = model(X)
            loss = criterion(yhat, ytrue)
            total += loss.item()
            loss.backward()
            optimizer.step()
        losses.append(total)
    
    return losses
    
    
    
D_in =1
H1=7
H2 = 3
D_out = 1
epochs = 10

model = NeuralNetwork(D_in, H1, H2, D_out)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#criterion = nn.CrossEntropyLoss()
#criterion = nn.CrossEntropyLoss()

X = torch.arange(-20, 20, 1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0] > -4) & (X[:, 0] < 4)] = 1.0

total_loss = train(X, Y, model, optimizer, crietrion, epochs)