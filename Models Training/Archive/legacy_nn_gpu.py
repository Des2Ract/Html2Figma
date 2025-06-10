import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib  # For saving encoders and scalers

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load dataset and remove rows with '-' in the tag column
df = pd.read_csv("../../feature_extraction/figma_dataset.csv")
df = df[~df['tag'].str.contains('-')]

# 2. Separate features and target
y = df["tag"]
X = df.drop(columns=["tag"])

# 3. Identify categorical and continuous columns
categorical_cols = ['type','parent_tag','parent_tag_html']
continuous_cols = [col for col in X.columns if col not in categorical_cols]

# Process categorical features with LabelEncoder
for col in categorical_cols:
    X[col] = X[col].astype(str)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    # If you need to save individual encoders, consider saving them in a dictionary.

# Fill missing values in continuous columns and scale them
X[continuous_cols] = X[continuous_cols].fillna(0)
scaler = StandardScaler()
X_continuous_scaled = scaler.fit_transform(X[continuous_cols])
joblib.dump(scaler, "scaler.pkl")

# Replace continuous columns in X with their scaled values
X_scaled = X.copy()
X_scaled[continuous_cols] = X_continuous_scaled

# 4. Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, "label_encoder.pkl")

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors and move to device
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# 7. Define the Neural Network Model with non-linear activations between linear layers
class TagClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(TagClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 128)           # Second hidden layer
        self.fc3 = nn.Linear(128, 256)            # Third hidden layer
        self.fc4 = nn.Linear(256, 512)            # Fourth hidden layer
        self.fc5 = nn.Linear(512, 512)            # Fifth hidden layer
        self.fc6 = nn.Linear(512, 512)            # Sixth hidden layer
        self.fc7 = nn.Linear(512, 256)            # Seventh hidden layer
        self.fc8 = nn.Linear(256, 128)           # Eighth hidden layer
        self.fc9 = nn.Linear(128, output_size)   # Output layer
        self.relu = nn.ReLU()                  # Non-linear activation

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        logits = self.fc9(x)  # No activation here: CrossEntropyLoss expects raw logits.
        return logits

# Initialize model and move to device
input_size = X_train_tensor.shape[1]
output_size = len(label_encoder.classes_)
model = TagClassifier(input_size, output_size).to(device)

# 8. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Internally applies softmax on logits
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Add batch processing for better GPU utilization
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop with batches
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "tag_classifier.pth")

# 10. Evaluation on the test set
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    y_pred = torch.argmax(outputs, dim=1).cpu().numpy()  # Move predictions back to CPU for sklearn metrics

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            labels=np.unique(y_test),
                            target_names=label_encoder.inverse_transform(np.unique(y_test))))