import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from Convnn import ChessNN
from excelsplit import load_dataset
from dataset import ChessDataset
import time
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessNN()
model.to(device)
print(f"Using device: {device}")

# Loss: multi-class cross entropy
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Data preparation
batch_size = 512  # Increased from 64 for better GPU utilization
print("Loading and preprocessing data...")
X_train, X_test, y_train, y_test = load_dataset("training_data.csv")

# Check if cached datasets exist
try:
    print("Trying to load preprocessed data from cache...")
    train_dataset = ChessDataset.load_from_cache("train_cache.pt")
    test_dataset = ChessDataset.load_from_cache("test_cache.pt")
    print("Loaded from cache successfully!")
except:
    print("Cache not found. Creating new datasets...")
    train_dataset = ChessDataset(X_train, y_train)
    test_dataset = ChessDataset(X_test, y_test)
    
    # Save cache for future runs
    train_dataset.save_cache("train_cache.pt")
    test_dataset.save_cache("test_cache.pt")
    print("Datasets cached for future use!")

# Data loader with optimizations
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=4,  # Use multiple CPU cores
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True  # Keep workers alive between epochs
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

num_epochs = 10

print(f"Starting training on {len(train_dataset)} samples...")
print(f"Batches per epoch: {len(train_loader)}")

for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0

    # Use tqdm for progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, (board_tensor, move_index) in enumerate(progress_bar):
        # Non-blocking transfer for better performance
        board_tensor = board_tensor.to(device, non_blocking=True)
        move_index = move_index.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(board_tensor)  # shape: (batch, 4672)
        
        loss = criterion(outputs, move_index)
        loss.backward()
        
        # Optional: gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        
        
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

    # Epoch summary
    end_time = time.time()
    epoch_time = end_time - start_time
    minutes = int(epoch_time // 60)
    seconds = int(epoch_time % 60)
    avg_loss = running_loss / len(train_loader)
    
    print(f"==> Epoch [{epoch+1}/{num_epochs}] completed")
    print(f"    Average Loss: {avg_loss:.4f}")
    print(f"    Time: {minutes}m {seconds}s")
    print(f"    Batches/sec: {len(train_loader)/epoch_time:.1f}")
    print("-" * 50)

    
print("Running quick evaluation...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for board_tensor, move_index in tqdm(test_loader, desc="Evaluating"):
        board_tensor = board_tensor.to(device, non_blocking=True)
        move_index = move_index.to(device, non_blocking=True)
        
        outputs = model(board_tensor)
        _, predicted = torch.max(outputs.data, 1)
        
        total += move_index.size(0)
        correct += (predicted == move_index).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), "chess_model_final.pth")
print("Model saved as 'chess_model_final.pth'")