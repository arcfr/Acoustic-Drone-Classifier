import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

FEATURES_DIR = "./data_preprocessed/mfccfeatures"
MODEL_CHECKPOINT_PATH = "bestwts/best_custom_cnn_drone_model.pth"
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 5         # Terminate loop if validation loss doesn't drop for 4 straight epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class DroneCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(DroneCNN, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 21, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.fc_block(x)
        return x

def load_and_partition_data():
    X = np.load(os.path.join(FEATURES_DIR, "X.npy"))
    y = np.load(os.path.join(FEATURES_DIR, "y.npy"))
    
    X = np.transpose(X, (0, 3, 1, 2))
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.10, random_state=42, stratify=y_train_val
    )
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, X_test, y_test

def train_and_validate(model, train_loader, val_loader, criterion, optimizer):
    print(f"Training initiated on execution device: {DEVICE}")
    print(f"Early Stopping Monitor Active (Patience: {PATIENCE} epochs)\n")
    
    # Tracking logs for tracking optimal performance snapshots
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = (correct_train / total_train) * 100
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = (correct_val / total_val) * 100
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] -> "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% || "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        # --- EARLY STOPPING & OPTIMAL SAVE SNAPSHOT LAYER ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
            patience_counter = 0  # Reset stagnation tracking
            print(f"   --> Validation loss improved. Optimal weights updated.")
        else:
            patience_counter += 1
            print(f"   --> No improvement. Patience counter: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print(f"\n[EARLY STOPPING TRIGGERED] Validation loss stagnated for {PATIENCE} epochs. Terminating loop.")
                break

def evaluate_testing_holdout(model, test_loader, X_test, y_test):
    print("\n================ TESTING PHASE EXECUTION ================")
    model.eval()
    
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.append(predicted.cpu().view(-1))
            all_true_labels.append(batch_y.view(-1))
            
    final_predictions = torch.cat(all_predictions).numpy()
    final_true_labels = torch.cat(all_true_labels).numpy()
    
    target_names = ["Light", "Medium", "Heavy", "No-Drone"]
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(final_true_labels, final_predictions, target_names=target_names))
    
    print("[CONFUSION MATRIX]")
    print(confusion_matrix(final_true_labels, final_predictions))

def main():
    train_loader, val_loader, test_loader, X_test, y_test = load_and_partition_data()
    
    model = DroneCNN(num_classes=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_and_validate(model, train_loader, val_loader, criterion, optimizer)
    
    # --- FINAL ROLLBACK PASS ---
    print(f"\nRolling back to the best recorded model weights from '{MODEL_CHECKPOINT_PATH}' for rigorous testing...")
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH))
    
    evaluate_testing_holdout(model, test_loader, X_test, y_test)

if __name__ == "__main__":
    main()