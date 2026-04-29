import torch
import torch.nn as nn
from tqdm import tqdm
import os


def train_model(model, dataloader, epochs, optimizer, criterion, save_path="best_model.pt"):
    save_dir = "models/{}".format(model.__class__.__name__)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_x, batch_y in progress_bar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)

            # outputs: [Batch * Seq_Len, Vocab]
            outputs_flat = outputs.transpose(1, 2).reshape(-1, dataloader.dataset.processor.vocab_size)
            # targets: [Batch * Seq_Len]
            targets_flat = batch_y.reshape(-1)

            loss = criterion(outputs_flat, targets_flat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

            model_path = os.path.join(save_dir, save_path)
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved ({model_path}) with loss: {best_loss}")
