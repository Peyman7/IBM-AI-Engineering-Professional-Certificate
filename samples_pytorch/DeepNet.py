import torch.nn as nn 
import torch.nn.functional as F

class DNet(nn.Module):
    def __init__(self, D_in, H, D_out, p):
        super(DNet, self).__init__() 
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, H)
        self.linear3 = nn.Linear(H, D_out)
        
        self.bn1 = nn.BatchNorm1d(H)
        self.bn2 = nn.BatchNorm1d(H)

        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = F.relu(self.drop(self.bn1(self.linear1(x))))
        x = F.relu(self.drop(self.bn2(self.linear2(x))))
        x = self.linear3(x)
        
        return x

        

class DeepNN(nn.Module):
    def __init__(self, layers, p):
        super(DeepNN, self).__init__() 
        self.hidden = nn.ModuleList()
        self.drop = p
        
        for input_size, output_size in zip(layers, layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.xavier_uniform_(linear.weight)
            self.hidden.append(linear))
            
    
    def forward(self, x):
        L = len(self.hidden)
        for l, linear_transform in zip(range(L), self.hidden):
            if l < L - 1: 
                x = F.relu(linear_transform(x))
                x = F.dropout(x, self.drop, training=self.training)
            else:
                x = linear_transform(x)
        return x 

def accuracy_score(all_preds, all_labels):
    predictions = torch.argmax(torch.stack(all_preds), dim=1)
    labels = torch.cat(all_labels)
    correct_preds = (predictions == labels).sum().item()
    accuracy = correct_preds / len(labels)
    return accuracy

def train(dataset, model, criterion, optimizer, epochs):
    results = {'train_loss': [], 'validation_loss': [], 'accuracy': []}
    
    # train loop 
    for epoch in range(epochs):
        total = 0
        for xbatch, ybatch in dataset['train_loader']:
            optimizer.zero_grad()
            yhat = model(xbatch)
            loss = criterion(yhat, ybatch)
            total += loss.item()
            loss.backward()
            optimizer.step() 
        results['train_loss'].append(total)
    
        # validation loop
        model.eval()    
        with torch.no_grad():
            all_preds = [] 
            all_labels = []
            val_loss = 0 
            for xval, yval in dataset['test_loader']:
                yhat = model(xval)
                loss = criterion(yhat, yval)
                val_loss += loss.item()
                all_preds.append(yhat)
                all_labels.append(yval)
            acc = accuracy_score(all_preds, all_labels)
            results['validation_loss'].append(val_loss)
            results['accuracy'].append(acc)
        print(f"Epoch {epoch+1}: Train Loss = {total:.4f}, Val Loss = {val_loss:.4f}, Accuracy = {acc:.2f}")

    return results 
    
layers = [2, 8, 6, 4, 3]
epochs = 20
learning_rate = 0.01
model = DeepNN(layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
results = train(dataset, model, criterion, optimizer, epochs)
