import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from Convnn import ChessNN
from excelsplit import load_dataset
from dataset import ChessDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessNN()
model.to(device)

# Loss: multi-class cross entropy
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# data preparation
batch_size = 64
X_train, X_test, y_train, y_test = load_dataset("training_data.csv")
train_dataset = ChessDataset(X_train, y_train)
test_dataset = ChessDataset(X_test, y_test)

# data loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 10  # you can change this

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (board_tensor, move_index) in enumerate(train_loader):
        board_tensor = board_tensor.to(device)
        move_index = move_index.to(device)
        
        optimizer.zero_grad()
        outputs = model(board_tensor)  # shape: (batch, 4672)
        
        loss = criterion(outputs, move_index)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        # --- print progress every 100 batches ---
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"==> Epoch [{epoch+1}/{num_epochs}] finished. Average Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "chess_model_final.pth")
print("Model saved as 'chess_model_final.pth'")
