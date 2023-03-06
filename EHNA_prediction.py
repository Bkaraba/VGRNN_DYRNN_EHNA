import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
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
        output, hn = self.rnn(x)
        hn = hn.squeeze()
        output = self.fc1(hn)
        return output
        
# load the data
with open('data/fb/adj_time_list.pickle', 'rb') as handle:
    adj_time_list = pickle.load(handle, encoding='latin1')

with open('data/fb/adj_orig_dense_list.pickle', 'rb') as handle:
    adj_orig_dense_list = pickle.load(handle, encoding='bytes')
   
# define the number of nodes and features in the dataset
num_nodes = adj_orig_dense_list[0].shape[0]
num_features = 1

# create a list to store the features for each time step
features_list = []


# loop through each time step and create a feature matrix for each time step
# loop through each time step and create a feature matrix for each time step
for adj_dense in adj_orig_dense_list:
    mask = adj_dense > 0
    #mask = mask.unsqueeze(-1)

    # convert the adjacency matrix to a binary feature matrix
    features = np.zeros((num_nodes, num_features))
    mask = np.expand_dims(mask, axis=-1)

    # create a new tensor with shape (num_nodes, num_nodes, num_features) and copy the values of mask into it
    # create a binary mask tensor
    desired_shape = (num_nodes, num_nodes, num_features)
    new_tensor = np.zeros(desired_shape)
    #print(mask.shape)
    #print(new_tensor.shape)

    #new_tensor[:, :, 0] = mask.squeeze(1)

    # assign the new tensor to features
    features = new_tensor

    features_list.append(features)
    #print(features_list)



# convert the feature list to a numpy array of shape (num_samples, num_timesteps, num_nodes, num_features)
features_array = np.array(features_list)
num_samples, num_timesteps = features_array.shape[:2]

# add an extra dimension to the features array to match the required format of the EHNA model
features_array = np.expand_dims(features_array, axis=4)

# create the adjacency matrix tensor of shape (num_samples, num_timesteps, num_nodes, num_nodes)
adj_tensor = np.zeros((num_samples, num_timesteps, num_nodes, num_nodes))
for i, adj_time in enumerate(adj_orig_dense_list):
    adj_tensor[i, :, :, :] = adj_time

# convert the adjacency matrix tensor to a binary tensor of shape (num_samples, num_timesteps, num_nodes, num_nodes)
adj_tensor[adj_tensor > 0] = 1

# create the label tensor of shape (num_samples, num_timesteps, num_nodes, num_features)
labels_tensor = features_array.copy()
labels_tensor[:, :-1, :, :] = labels_tensor[:, 1:, :, :]

#split the dataset into training and test sets
train_idx = int(0.8 * num_samples)
x_train, x_test = features_array[:train_idx], features_array[train_idx:]
y_train, y_test = labels_tensor[:train_idx], labels_tensor[train_idx:]
adj_train, adj_test = adj_tensor[:train_idx], adj_tensor[train_idx:]

#Create tensor datasets
train_data = torch.utils.data.TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
test_data = torch.utils.data.TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

#Define dataloaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

#Define training parameters
num_temporal_dimensions = adj_orig_dense_list.shape[0]
input_size = num_nodes + num_temporal_dimensions
hidden_size = 64 # Change this value based on your preference
output_size = 1 # Change this value based on your task

learning_rate = 0.01
num_epochs = 50

#model
model = EHNA(input_size,hidden_size,output_size)


#Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Train EHNA model
for epoch in range(num_epochs):
  running_loss = 0.0
  for inputs, labels in train_loader:
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

# Evaluate EHNA model on test dataset
with torch.no_grad():
    test_inputs, test_labels = next(iter(test_loader))
    predictions = model(test_inputs)
    binary_preds = (predictions > 0.5).float()
    f1 = f1_score(test_labels.numpy(), binary_preds.numpy(), average='micro')
    precision = precision_score(test_labels.numpy(), binary_preds.numpy(), average='micro')
    recall = recall_score(test_labels.numpy(), binary_preds.numpy(), average='micro')
    test_loss = criterion(predictions, test_labels)

print('Epoch: {}, Loss: {:.4f}, Test Loss: {:.4f}, F1 score: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'.format(epoch+1, running_loss, test_loss, f1, precision, recall)) 
#save the trained model
torch.save(model.state_dict(), 'models/EHNA_model.pth')
