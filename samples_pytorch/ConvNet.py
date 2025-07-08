import torch.nn as nn 
import torch.nn.functional as F 


class ConvNet(nn.Module):
    def __init__(self, out1, out2, input_size=(1, 28, 28)):
        super(ConvNet, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out1, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.cnn2 = nn.Conv2d(in_channels=out1, out_channels=out2, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        flattened_size = __get_flattened_size(input_size)
        self.fc1 = nn.Linear(flattened_size, 10)
       
    def __get_flattened_size(self, x_shape):
        with torch.no_grad():
            x = torch.zeros(1, *x_shape)
            x = self.maxpool1((F.relu(self.self.cnn1)))
            x = self.maxpool2((F.relu(self.self.cnn2))
            return x.view(1, -1).size(1)

            
    
    def forward(self, x):
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.maxpool1(x) 
        
        x = self.cnn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        return x 
        

def accuracy_score(all_preds, all_labels):
    predictions = torch.argmax(torch.stack(all_preds), dim=1)
    labels = torch.cat(all_labels)
    correct_preds = (predictions == labels).sum().item()
    accuracy = correct_preds / len(labels)
    return accuracy

def train(model, criterion, optimizer, dataset, epochs):
    
    results = {'train_loss':[], 'validation_loss': [], 'accuracy': []}
    # Train Loop 
    model.train()
    for epoch in range(epochs):
        total = 0
        for xbatch, ybatch in dataset['train_set']:
            optimizer.zero_grad() 
            yhat = model(xbatch)
            loss = criterion(yhat, y)
            total += loss.item() 
            
            loss.backward()
            optimizer.step() 
        results['traing_loss'].append(total)
        
        # validation loop
        val_loss = 0
        all_preds = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for xval, yval in dataset['validation_set']:
                yhat = model(xval)
                loss = citerion(yhat, yval)
                val_loss += loss.item() 
                all_preds.append(yhat)
                all_labels.append(yval)
            
            acc = accuracy_score(all_preds, all_labels)
            results['validation_loss'].append(val_loss)
            results['accuracy'].append(acc)
        
        print(f"Epoch {epoch+1}: Train Loss = {total:.4f}, Val Loss = {val_loss:.4f}, Accuracy = {acc:.2f}")
    
    return results 


out1= 16
out2 = 32 
input_size = (1, 28, 28)
model = ConvNet(out1, out2, input_size)


transform = torch.transforms.Compose([transforms.Resize((input_size[1], input_size[2])), transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root=',.data', train=True, download=True, transform=transform)
validation_dataset = torchvision.datasets.MNIST(root=',.data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
val_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)

dataset = {'train_set': train_loader, 'validation_set': val_loader}

epochs = 5 
criterion = nn.CrossEntropyLoss() 
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

results = train(model, criterion, optimizer, dataset, epochs)