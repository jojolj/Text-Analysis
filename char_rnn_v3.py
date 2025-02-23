import torch
import torch.nn as nn
import numpy as np

# Step 1: Data Preparation
def read_and_one_hot_encode(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Create a dictionary to convert characters to integer indexes
    chars = sorted(list(set(text)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}

    # One-hot encode
    data_encoded = [char_to_int[ch] for ch in text]
    data_one_hot = torch.zeros(len(data_encoded), len(chars))
    data_one_hot[range(len(data_encoded)), data_encoded] = 1

    return data_one_hot, char_to_int, int_to_char

# Load and prepare data
file_path = 'D:\\MasterStudies\\PatternRecognitionandMachineLearning\\Exercises\\abcde_edcba.txt'
data_one_hot, char_to_int, int_to_char = read_and_one_hot_encode(file_path)


# Parameters for the model
input_size = len(char_to_int) 
state_size = 128  # Defining the size of the internal state
output_size = len(char_to_int) 

# Data preparation
X = []
Y = []
state_inputs = []

for i in range(len(data_one_hot) - 1):
    X.append(data_one_hot[i])
    Y.append(data_one_hot[i + 1])
    # Initially, the state input will be zeros
    state_inputs.append(torch.zeros((1, state_size)))

X = torch.stack(X)
Y = torch.stack(Y)
state_inputs = torch.stack(state_inputs).squeeze(1)  # Remove the extra dimension


# Step 2: Model Setup
class StatefulMLP(nn.Module):
    def __init__(self, input_size, state_size, output_size):
        super(StatefulMLP, self).__init__()
        self.state_size = state_size  # Define state_size as an instance variable
        self.fc1 = nn.Linear(input_size + state_size, state_size)
        self.fc2 = nn.Linear(state_size, output_size)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)  # Concatenate input and state
        h_next = torch.tanh(self.fc1(combined))
        y = self.fc2(h_next)
        return y, h_next

    def init_state(self, batch_size=1):
        return torch.zeros(batch_size, self.state_size)


# Instantiate the model
model = StatefulMLP(input_size, state_size, output_size)
#%%
# Assuming the StatefulMLP, X, Y, state_inputs are already defined as shown above

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100  # Set the number of epochs for training
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = 0
    h = model.init_state()  # Initialize the internal state
    
    for i in range(len(X)):
        y_pred, h = model(X[i].unsqueeze(0), h)  # Forward pass, unsqueeze to add batch dimension
        loss += criterion(y_pred, Y[i].unsqueeze(0).max(dim=1)[1])  # Accumulate loss

    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss {loss.item() / len(X)}')  # Print average loss per character
#%%
def predict(model, char_to_int, int_to_char, initial_chars='ab', sequence_length=50):
    # Initialize the state and the sequence
    h = model.init_state()
    sequence = initial_chars
    
    # Convert initial characters to one-hot encoding
    for char in initial_chars[:-1]:  # Process all but the last character to update the state
        x = torch.zeros((1, input_size))
        x[0, char_to_int[char]] = 1
        _, h = model(x, h)
    
    # Now predict the next characters
    last_char = initial_chars[-1]
    for _ in range(sequence_length - len(initial_chars)):
        x = torch.zeros((1, input_size))
        x[0, char_to_int[last_char]] = 1
        y_pred, h = model(x, h)
        last_char = int_to_char[y_pred.argmax(1).item()]
        sequence += last_char
    
    return sequence

# Generate a sequence starting with 'ab'
generated_sequence = predict(model, char_to_int, int_to_char, initial_chars='ab', sequence_length=50)
print(generated_sequence)

















