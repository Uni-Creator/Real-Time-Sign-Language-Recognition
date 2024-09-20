import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Define parameters
DATA_PATH = 'Data/'  # Set your data path here
actions = ['nothing', 'hello', 'thanks', 'iloveyou']  # List your actions here
sequence_length = 30  # Set the length of each sequence

# Prepare label mapping
label_map = {label: num for num, label in enumerate(actions)}

# Load sequences and labels
sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Convert to numpy arrays
X = np.array(sequences)  # Shape: (180, 30, 1662)
y = np.array(labels)      # Shape: (180,)

# Convert labels to one-hot encoding
y = np.eye(len(actions))[y]  # One-hot encoding

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Convert to PyTorch tensors and move to the GPU if available
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size=hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True)
        self.lstm3 = nn.LSTM(128, 64, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.dropout(x[:, -1, :])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# Initialize model, loss function, optimizer, and scheduler
model = LSTMModel(input_size=1662, hidden_size=256, output_size=len(actions)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

# Hyperparameters
num_epochs = 2000
patience = 20  # Early stopping patience

# Early stopping and best model saving
best_val_loss = float('inf')
early_stop_counter = 0
best_model_path = 'best_lstm_model.h5'

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels.argmax(dim=1))  # Compute loss
        loss.backward()  # Backward pass and optimization
        optimizer.step()  # Update weights
        running_loss += loss.item()

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)  # Move to GPU
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_labels.argmax(dim=1)).item()
    val_loss /= len(val_loader)

    # scheduler.step(val_loss)  # Adjust learning rate based on validation loss

    # Print statistics
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

    # Early stopping and saving best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}'")
        break

# Load the best model
model.load_state_dict(torch.load(best_model_path, weights_only=True))
model = model.to(device)

# Evaluate the model on the test set and print probabilities
model.eval()
y_pred = []
y_true = []

print("Class predictions with probabilities:\n")
with torch.no_grad():
    for test_inputs, test_labels in zip(X_test, y_test):
        test_inputs = test_inputs.unsqueeze(0).to(device)  # Move to GPU and add batch dimension
        test_outputs = model(test_inputs)  # Forward pass

        # Get the predicted class and its probability
        probabilities = test_outputs.squeeze().cpu().numpy()  # Convert to numpy for easier reading and move back to CPU
        predicted_class = np.argmax(probabilities)

        # Print the predicted probabilities for each class
        for i, action in enumerate(actions):
            print(f"{action}: {probabilities[i]:.4f}")

        print(f"Predicted: {actions[predicted_class]}, True: {actions[test_labels.argmax().item()]}\n")

        y_pred.append(predicted_class)
        y_true.append(test_labels.argmax().item())

# Now print the confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=actions, yticklabels=actions)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("Model trained and saved successfully...")
