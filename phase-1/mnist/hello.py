import torch                          # Main PyTorch library
import torch.nn as nn                # Neural network layers and loss functions
from torch.utils.data import DataLoader   # Helps load data in batches
from torchvision import datasets, transforms  # MNIST dataset and image transforms
import matplotlib.pyplot as plt      # For plotting images
import numpy as np                   # Numerical utilities (not strictly required here)

# -----------------------------
# 1. Config
# -----------------------------
batch_size = 64                      # Number of images processed at once
epochs = 5                          # Number of full passes through training data
learning_rate = 1e-3                # Step size for optimizer updates
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use GPU if available, otherwise use CPU

print("Using device:", device)      # Show which device is being used

# -----------------------------
# 2. Load MNIST dataset
# -----------------------------
transform = transforms.Compose([    
    transforms.ToTensor(),          # Convert PIL image to tensor with shape [C, H, W]
    transforms.Normalize((0.1307,), (0.3081,))  
    # Normalize using MNIST mean and std so training is more stable
])

train_dataset = datasets.MNIST(
    root="./data",                  # Folder where dataset will be stored
    train=True,                     # Load training split
    transform=transform,            # Apply transform to each image
    download=True                   # Download dataset if missing
)

test_dataset = datasets.MNIST(
    root="./data",                  # Same storage folder
    train=False,                    # Load test split
    transform=transform,            # Apply same preprocessing
    download=True                   # Download if needed
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Wrap training data in a DataLoader
# batch_size=64 means 64 images per batch
# shuffle=True randomizes order each epoch to improve training

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Test data loader
# shuffle=False because order does not matter during evaluation

# -----------------------------
# 3. Build a simple 2-layer NN
#    (Flatten -> Linear -> ReLU -> Linear)
# -----------------------------
class SimpleNN(nn.Module):          # Define a custom neural network class
    def __init__(self):
        super().__init__()          # Initialize parent nn.Module

        self.flatten = nn.Flatten() # Flattens 28x28 image into 784-length vector
        self.fc1 = nn.Linear(28 * 28, 128)
        # First fully connected layer: 784 inputs -> 128 hidden units

        self.relu = nn.ReLU()       # Nonlinear activation function
        self.fc2 = nn.Linear(128, 10)
        # Second fully connected layer: 128 hidden units -> 10 output classes

    def forward(self, x):
        x = self.flatten(x)         # Convert [B, 1, 28, 28] to [B, 784]
        x = self.fc1(x)             # Apply first linear layer
        x = self.relu(x)            # Apply ReLU activation
        x = self.fc2(x)             # Output raw scores (logits) for 10 digits
        return x                    # Return logits

model = SimpleNN().to(device)       # Create model and move it to CPU or GPU

criterion = nn.CrossEntropyLoss()   # Standard loss for multiclass classification
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Adam optimizer updates model weights using gradients

# -----------------------------
# 4. Train
# -----------------------------

initial_weights = model.fc1.weight.data.clone().cpu()

for epoch in range(epochs):         # Repeat training for the chosen number of epochs
    model.train()                   # Put model in training mode
    running_loss = 0.0              # Track total loss for this epoch

    for images, labels in train_loader:
        # Get one batch of images and labels from training data

        images, labels = images.to(device), labels.to(device)
        # Move batch to same device as model

        optimizer.zero_grad()
        # Clear old gradients from previous batch

        outputs = model(images)
        # Run forward pass: model predicts logits for each image

        loss = criterion(outputs, labels)
        # Compare predictions with correct labels and compute loss

        loss.backward()
        # Backpropagation: compute gradients of loss w.r.t. parameters

        optimizer.step()
        # Update parameters using gradients

        running_loss += loss.item()
        # Add scalar loss value for monitoring

    avg_loss = running_loss / len(train_loader)
    # Compute average loss over all batches in this epoch

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    # Print training progress

# -----------------------------
# 5. Evaluate on test set
# -----------------------------
model.eval()                        # Put model in evaluation mode
correct = 0                         # Count correct predictions
total = 0                           # Count total predictions

all_images = []                     # Store all test images for visualization later
all_labels = []                     # Store all true labels
all_preds = []                      # Store all predicted labels

with torch.no_grad():               # Disable gradient computation to save memory/speed
    for images, labels in test_loader:
        # Loop through test batches

        images, labels = images.to(device), labels.to(device)
        # Move data to same device as model

        outputs = model(images)
        # Get predicted logits

        preds = outputs.argmax(dim=1)
        # Pick class with highest score as prediction

        correct += (preds == labels).sum().item()
        # Count how many predictions were correct in this batch

        total += labels.size(0)
        # Add batch size to total number of examples

        all_images.append(images.cpu())
        # Save images on CPU for later plotting

        all_labels.append(labels.cpu())
        # Save true labels

        all_preds.append(preds.cpu())
        # Save predicted labels

test_accuracy = 100.0 * correct / total
# Convert accuracy into percentage

print(f"Test Accuracy: {test_accuracy:.2f}%")
# Print final test accuracy

all_images = torch.cat(all_images, dim=0)
# Merge list of image batches into one tensor

all_labels = torch.cat(all_labels, dim=0)
# Merge all true labels into one tensor

all_preds = torch.cat(all_preds, dim=0)
# Merge all predicted labels into one tensor

# -----------------------------
# 6. Visualize predictions
#    Show some correct and incorrect examples
# -----------------------------
correct_idx = (all_preds == all_labels).nonzero(as_tuple=True)[0]
# Get indices where prediction matches true label

incorrect_idx = (all_preds != all_labels).nonzero(as_tuple=True)[0]
# Get indices where prediction is wrong

def denormalize(img):
    # Reverse the normalization so image looks normal when plotted
    return img * 0.3081 + 0.1307

def show_examples(indices, title, n=8):
    plt.figure(figsize=(14, 4))     # Create figure
    plt.suptitle(title, fontsize=14) # Overall figure title

    for i in range(min(n, len(indices))):
        # Loop over first n selected examples

        idx = indices[i].item()
        # Convert tensor index to Python integer

        img = denormalize(all_images[idx]).squeeze().numpy()
        # Undo normalization
        # Remove channel dimension with squeeze()
        # Convert tensor to NumPy array for matplotlib

        plt.subplot(2, 4, i + 1)
        # Create a 2x4 grid and place image in slot i+1

        plt.imshow(img, cmap="gray")
        # Display image in grayscale

        plt.title(f"Pred: {all_preds[idx].item()} | True: {all_labels[idx].item()}")
        # Show predicted label and actual label above image

        plt.axis("off")
        # Hide axes for cleaner display

    plt.tight_layout()
    # Adjust spacing between subplots

    plt.show()
    # Display figure

show_examples(correct_idx, "Correct Predictions", n=8)
# Show 8 correctly classified examples

show_examples(incorrect_idx, "Incorrect Predictions", n=8)
# Show 8 incorrectly classified examples

import matplotlib.pyplot as plt

# Get weights from first layer
weights = model.fc1.weight.data.cpu()  # shape: [128, 784]

# Pick a few neurons to visualize
num_neurons_to_show = 8

plt.figure(figsize=(12, 4))

for i in range(num_neurons_to_show):
    w = weights[i]                  # shape: [784]
    w_img = w.view(28, 28)          # reshape to image

    plt.subplot(2, 4, i + 1)
    plt.imshow(w_img, cmap="seismic")  # red/blue shows positive/negative weights
    plt.colorbar()
    plt.title(f"Neuron {i}")
    plt.axis("off")

plt.suptitle("First Layer Neuron Weights (What they 'look for')")
plt.tight_layout()
plt.show()

def plot_weights(weights, title):
    plt.figure(figsize=(12, 4))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(weights[i].view(28, 28), cmap="seismic")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

plot_weights(initial_weights, "Before Training")
plot_weights(model.fc1.weight.data.cpu(), "After Training")