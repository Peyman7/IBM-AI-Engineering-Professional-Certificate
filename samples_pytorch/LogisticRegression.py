import torch 
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class MyDataset(Dataset):

    def __init__(self):
        self.x = torch.arange(-3, 3, 0.01).view(-1, 1)
        self.y = -3 * self.x + torch.randn(self.x.size())
        self.len = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
    def __len__(self):
        return self.len 

class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

        

class Trainer:

    def __init__(self, model, learning_rate, iterations, trainloader, valloader):
        self.model = model 
        self.learning_rate = learning_rate
        self.iterations = iterations 
        self.trainloader = trainloader 
        self.valloader = valloader 
        self.loss_fn = nn.BCELoss() 
        self.optimizer = optim.SGD(self.model.parameters(), self.learning_rate)
        
    def train(self):
        for epoch in range(self.iterations):
            for x, y in self.trainloader: 
                self.optimizer.zero_grad()
                yhat = self.model(x)
                loss = self.loss_fn(yhat, y)
                loss.backward() 
                self.optimizer.step() 
    
    def validate(self):
        total_loss = 0
        with torch.no_grad():
            for xval, yval in self.valloader:
                yhat = self.model(xval)
                loss = self.loss_fn(yhat, yval)
                total_loss += loss.item()
        return total_loss 
        
iterations = 4
lr = 0.01
w = torch.tensor(-10.0, requires_grad=True)
b = torch.tensor(-15.0, requires_grad=True)

dataset = MyDataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])
trainloader = DataLoader(train_data, batch_size=1)
valloader = DataLoader(val_data, batch_size=1)


learning_rates = [0.001, 0.01, 0.05, 0.1]
models = []
val_losses = []
iterations = 4

def process_train_validation(learning_rates, iterations, trainloader, valloader):
    models = []
    val_losses = []
    for lr in learning_rates:
    
        model = LogisticRegression(1, 1)
        trainer = Trainer(model, lr, iterations, trainloader, valloader)
        trainer.train() 
        val_loss = trainer.validate()
        models.append(model)
        val_losses.append(val_loss)
    return models, val_losses


models, val_losses = process_train_validation(learning_rates, iterations, trainloader, valloader)
best_idx = torch.argmin(torch.tensor(val_losses))
best_model = models[best_idx]

print(f"✅ Best learning rate: {learning_rates[best_idx]}, Validation loss: {val_losses[best_idx]:.4f}")

# Plot results
x_vals = dataset.x
y_vals = dataset.y
with torch.no_grad():
    y_pred = best_model(x_vals)

plt.scatter(x_vals.numpy(), y_vals.numpy(), label='Data', alpha=0.5)
plt.plot(x_vals.numpy(), y_pred.numpy(), color='red', label='Best Fit')
plt.title("Linear Regression with Best Learning Rate")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()