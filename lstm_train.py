import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lstm_data_prep import load_all_instances
from lstm_data_prep import read_flow_shop_data, generate_labels_with_neh, FlowShopDataset
from lstm_model import LSTMSequencer

# Parametry
filename = "flowshop_100jobs_20machines.txt"
# batch_size = 4
batch_size = 16
hidden_size = 128
num_layers = 2
num_epochs = 1000
learning_rate = 0.001

# Early stopping – parametry
early_stopping_patience = 150
min_loss_delta = 0.0001  # minimalna poprawa

# Wczytaj dane
num_jobs, num_machines, processing_times = read_flow_shop_data(filename)
labels = generate_labels_with_neh(processing_times)
dataset = load_all_instances("data/flowshop_instances")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Inicjalizacja modelu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMSequencer(
    input_size=1,
    hidden_size=hidden_size,
    output_size=num_jobs,
    num_layers=num_layers
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Early stopping – zmienne
best_loss = float('inf')
patience_counter = 0

# Trening
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.unsqueeze(-1).to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoka [{epoch + 1}/{num_epochs}], Strata: {avg_loss:.4f}")

    # Early stopping logika
    if best_loss - avg_loss > min_loss_delta:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), "lstm_model.pth")  # Zapis tylko przy poprawie
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print(f"\n⏹️ Trening zatrzymany wcześniej po ep. {epoch+1} – brak poprawy przez {early_stopping_patience} epok.")
        break

print(f"\n✅ Model zapisany jako lstm_model.pth (najlepsza strata: {best_loss:.4f})")
