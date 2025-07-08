import torch 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class my_dataset(Dataset):

    def __init__(self):
        self.x = torch.arange(-3, 3, 0.01).view(-1, 1)
        self.y = -3 * self.x + torch.randn(self.x.size())
        self.len = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.len 



class LinearRegression:
    def __init__(self, w, b, lr, iterations, trainloader):
        self.w = w 
        self.b = b 
        self.lr = lr 
        self.iterations = iterations 
        self.trainloader = trainloader 
    
    def forward(self, x):
        return self.w * x + self.b 
    def criterion(self, y_true, y_pred):
        return torch.mean(( y_true - y_pred)**2)
    def train(self):
        LOSS = []
        for epoch in range(self.iterations):
            total = 0
            for x, y in self.trainloader: 
                yhat = self.forward(x)
                loss = self.criterion(y, yhat)
                                
                loss.backward() 
                with torch.no_grad():
                    self.w -= self.lr * self.w.grad 
                    self.w.grad.zero_() 
                
                    self.b -= self.lr * self.b.grad 
                    self.b.grad.zero_() 
                
                total +=loss.item() 
            LOSS.append(total)
        
        return LOSS 
        
iterations = 4
lr = 0.01
w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-15.0, requires_grad=True)

my_data = my_dataset()
print("The length of dataset: ", len(my_data))
trainloader = DataLoader(dataset=my_data, batch_size=1)

model = LinearRegression(w, b, lr, iterations, trainloader)                
losses = model.train()


# Plot original data
x_vals = my_data.x
y_vals = my_data.y

# Plot model predictions
with torch.no_grad():
    y_pred = model.forward(x_vals)

plt.scatter(x_vals.numpy(), y_vals.numpy(), label='Data', alpha=0.5)
plt.plot(x_vals.numpy(), y_pred.numpy(), color='red', label='Fitted Line')
plt.title("Linear Regression with SGD")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()