import torch


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    epochs=50,
    device="cpu",
    gradient_clipping=1.0,
    early_stopping_patience=5,
    model_name="Model",
):
    """
    Train a PyTorch model with best practices.

    Args:
        model (torch.nn.Module): Model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (torch.nn.Module): Loss function (e.g., MSELoss).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., Adam, SGD).
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        epochs (int): Number of training epochs.
        device (str): Device to run training on (e.g., "cuda", "cpu").
        gradient_clipping (float): Max norm for gradient clipping.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
        model_name (str): Name of the model for logging purposes.

    Returns:
        Tuple[list, list]: Training and validation losses per epoch.
    """
    model.to(device)
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation Phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Adjust learning rate
        if scheduler:
            scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} - {model_name}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered for {model_name}.")
            break

    # Load the best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses
