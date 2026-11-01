DeepLOB PyTorch Replication
A PyTorch implementation of DeepLOB for Limit Order Book (LOB) price movement prediction.

📌 Features
Data: High-frequency tick data processing using Polars.

Model: CNN-LSTM architecture (DeepLob) for 3-class classification.

Training: Early stopping, LR scheduler, and best-model checkpointing.

Evaluation: TorchMetrics (Macro F1/Precision/Recall) and scikit-learn for confusion matrix.

🛠 Installation
bash
git clone https://github.com/your_username/deeplob-pytorch.git
cd deeplob-pytorch
pip install torchinfo torchmetrics polars pandas numpy scikit-learn pyyaml
🚀 Usage
Place your raw data in the data/ directory.

Run the training pipeline (example from Jupyter Notebook):

python
from src.model.model_set_up import DeepLob
from src.train_and_evaluation.train_and_evaluate import train_and_evaluate, test_model
from torch.utils.data import DataLoader

# 1. Prepare Data
batch_size, h, num_classes, epochs = 64, 0, 3, 50
# dataset_train, dataset_val, dataset_test = ... (see notebook for details)
train_loader = DataLoader(dataset_train, batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size, shuffle=False)

# 2. Initialize Model & Optimizer
model = DeepLob(y_len=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# 3. Train
best_path = BASE_DIR / 'model_trained/best_model.pt'
train_and_evaluate(model, train_loader, val_loader, criterion,
                   optimizer, n_epochs=epochs, device='cuda', 
                   scheduler=scheduler, early_stop=False, 
                   patience=5, path=best_path)

# 4. Test
best_model = torch.load(best_path, weights_only=True)
model.load_state_dict(best_model)
test_model(model, test_loader, criterion, device='cuda')
📊 Results
Accuracy: 0.XXXX (fill in your results)

Macro F1: 0.XXXX

Confusion Matrix: (paste your generated image here)

Based on: DeepLOB: Deep Convolutional Neural Networks for Limit Order Books (Zhang et al., 2019).

text
