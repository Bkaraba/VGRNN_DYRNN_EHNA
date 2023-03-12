import numpy as np
import pickle
import torch
import time
#from torch._C import double
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import TensorDataset, DataLoader

# Define EHNA model architecture
class EHNA(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EHNA, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity='relu', batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.float() 
        output, hn = self.rnn(x)
        hn = hn.squeeze()
        output = self.fc1(hn)
        return output

def calculate_metrics(model, dataloader):
    y_true = []
    y_pred = []

    for inputs, labels in dataloader:
        # forward pass
        outputs = model(inputs)

        # convert outputs and labels to numpy arrays
        outputs = outputs.detach().numpy()
        labels = labels.detach().numpy()

        # append true and predicted labels to lists
        y_true.extend(labels.flatten())
        y_pred.extend(outputs.flatten())

    # convert true and predicted labels to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # threshold predicted labels
    y_pred = np.where(y_pred > 0, 1, 0)

    # calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    return precision, recall, f1

# load the data
with open('data/fb/adj_time_list.pickle', 'rb') as handle:
    adj_time_list = pickle.load(handle, encoding='latin1')

with open('data/fb/adj_orig_dense_list.pickle', 'rb') as handle:
    adj_orig_dense_list = pickle.load(handle, encoding='bytes')

# convert adj_time_list and adj_orig_dense_list to PyTorch tensors
#adj_time_list = torch.tensor(adj_time_list)
adj_orig_dense_list = torch.stack(adj_orig_dense_list)

# define the number of nodes and features in the dataset
num_nodes = adj_orig_dense_list[0].shape[1]
num_features = 663

# create a list to store the features for each time step
features_list = []

for adj_dense in adj_orig_dense_list:
    mask = adj_dense > 0
    features = np.zeros((num_nodes, num_features))
    features = torch.tensor(features)
    features_list.append(features)

# convert the features list to a PyTorch tensor
features_tensor = torch.stack(features_list)

# create a target tensor that has the same shape as the output of the model
target = adj_orig_dense_list.reshape(-1, num_nodes * num_nodes)

# split the dataset into training and validation sets
train_size = int(0.8 * len(features_tensor))
val_size = len(features_tensor) - train_size

train_features = features_tensor[:train_size]
train_target = target[:train_size]

val_features = features_tensor[train_size:]
val_target = target[train_size:]

# create PyTorch dataloaders for the training and validation sets
train_dataset = TensorDataset(train_features, train_target)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(val_features, val_target)
val_loader = DataLoader(val_dataset, batch_size=32)

# initialize the EHNA model and define the loss function and optimizer
ehna_model = EHNA(input_size=num_features, hidden_size=128, output_size=num_nodes * num_nodes)
criterion = nn.MSELoss()
optimizer = optim.Adam(ehna_model.parameters(), lr=0.001)

# set device to CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# set number of epochs
inum_epochs = 20

# initialize lists to store loss and accuracy for each epoch
train_losses = []
val_losses = []
accuracies = []


# train the model
for epoch in range(num_epochs):
    start_time = time.time()
    # set model to training mode
    ehna_model.train()

    # initialize running loss
    running_loss = 0.0

    # loop over training set in batches
    for i, (inputs, labels) in enumerate(train_loader):
        # move inputs and labels to device
        #inputs = inputs.to(device)
        #labels = labels.to(device)
        inputs = inputs.double()
        #inputs.to(torch.double)
        #print(inputs.dtype)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = ehna_model(inputs.double())

        # compute loss
        loss = criterion(outputs, labels)

        # backward pass and optimize
        loss.backward()
        optimizer.step()

        # update running loss
        running_loss += loss.item()

    # append training loss for this epoch
    train_losses.append(running_loss / len(train_loader))

    # switch model to evaluation mode and disable gradient calculation
    ehna_model.eval()
    with torch.no_grad():
        # initialize validation loss and accuracy
        val_loss = 0.0
        correct = 0
        total = 0

        # loop over validation set in batches
        for inputs, labels in val_loader:
            # move inputs and labels to device
            #inputs = inputs.to(device)
            #labels = labels.to(device)

            # forward pass
            outputs = ehna_model(inputs)

            # compute loss
            val_loss += criterion(outputs, labels).item()

        # append validation loss for this epoch
        val_losses.append(val_loss / len(val_loader))

    # calculate precision, recall, and F1 score
    precision, recall, f1 = calculate_metrics(ehna_model, val_loader)

    # append accuracy for this epoch
    accuracies.append(f1)
    end_time = time.time()
    time_cost = end_time - start_time
    # print metrics for this epoch
    print('epoch:', epoch+1)
    print('training loss:', train_losses[-1])
    print('validation loss:', val_losses[-1])
    print('f1:', f1)
    print("precision:", precision)
    print('recall:', recall)
    print('time_cost = {:2f} seconds'.format(time_cost))

