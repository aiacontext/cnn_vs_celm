# traditional_trainer.py

import time
import torch.nn as nn
import torch.optim as optim


def train_traditional_cnn(model, train_loader, device, epochs=5, lr=0.001):
    """Treina uma CNN tradicional usando backpropagation"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    start_time = time.time()
    training_stats = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        training_stats.append({
            'epoch': epoch + 1,
            'loss': epoch_loss
        })

        print(f'Época {epoch + 1}, Loss: {epoch_loss:.4f}')

    training_time = time.time() - start_time

    return {
        'model': model,
        'training_time': training_time,
        'training_stats': training_stats
    }