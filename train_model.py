import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Load dataset
data_dir = "dataset"
all_data = []
all_labels = []
label_map = {}

for i, filename in enumerate(os.listdir(data_dir)):
    label = filename.split(".")[0]
    label_map[i] = label
    df = pd.read_csv(os.path.join(data_dir, filename), header=None)
    all_data.append(df.values)
    all_labels += [i] * len(df)

X = np.vstack(all_data).astype(np.float32)
y = np.array(all_labels).astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# Define model
class GestureNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureNet(X.shape[1], len(label_map)).to(device)

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save model and labels
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/sign_language_model.pth")

with open("model/labels.txt", "w") as f:
    for i in label_map:
        f.write(f"{i},{label_map[i]}\n")
