import logging

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Set logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def _forward_pass(model, X_batch, device):
    """
    Handle arbitrary input formats:
    If X_batch is a tuple or list, unpack and pass as separate args.
    Otherwise, pass as a single argument.
    """
    if isinstance(X_batch, (tuple, list)):
        X_batch = [x.to(device) for x in X_batch]
        outputs = model(*X_batch)
    else:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
    return outputs


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
    use_mixed_precision=True,
):
    """
    A unified training loop for any PyTorch model. This loop:
    - Handles regression or classification based on criterion and model outputs.
    - Supports early stopping, gradient clipping, LR scheduling.
    - Uses mixed precision if enabled.
    - Works for single or multiple input modalities (unimodal/multimodal).

    Args:
        model (nn.Module): PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (optional): Learning rate scheduler that implements step(val_loss).
        epochs (int): Max number of training epochs.
        device (str): "cpu" or "cuda".
        gradient_clipping (float): Gradient norm clipping value.
        early_stopping_patience (int): Patience for early stopping based on val_loss.
        model_name (str): Name of the model for logging.
        use_mixed_precision (bool): Use AMP for faster training on GPU.

    Returns:
        (train_losses, val_losses): Lists of training and validation losses per epoch.
    """
    model.to(device)
    scaler = GradScaler(enabled=use_mixed_precision)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in tqdm(range(epochs)):
        model.train()
        running_train_loss = 0.0

        for batch in train_loader:
            # Expecting batch to be (X, y)
            # If multiple inputs, batch may be (X1, X2, ..., y)
            # Last element is assumed to be y
            *X_batch, y_batch = batch
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_mixed_precision):
                outputs = _forward_pass(model, X_batch, device)
                outputs = (
                    outputs.squeeze(dim=-1)
                    if outputs.dim() > 1 and outputs.size(-1) == 1
                    else outputs
                )
                loss = criterion(outputs, y_batch)

            scaler.scale(loss).backward()
            # Gradient Clipping
            if gradient_clipping is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item() * y_batch.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad(), autocast(enabled=use_mixed_precision):
            for batch in val_loader:
                *X_batch, y_batch = batch
                y_batch = y_batch.to(device)

                outputs = _forward_pass(model, X_batch, device)
                outputs = (
                    outputs.squeeze(dim=-1)
                    if outputs.dim() > 1 and outputs.size(-1) == 1
                    else outputs
                )
                val_loss = criterion(outputs, y_batch)
                running_val_loss += val_loss.item() * y_batch.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # Learning Rate Scheduler Step
        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        # Early Stopping Check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logging.info(
                    f"Early stopping triggered for {model_name} at epoch {epoch+1}."
                )
                break

    # Restore best model weights
    if best_model_state:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses


def train_multimodal_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    epochs=10,
    device="cpu",
    gradient_clipping=1.0,
    early_stopping_patience=5,
    model_name="Perturbinator",
    use_mixed_precision=True,
):
    model.to(device)
    scaler = GradScaler(enabled=use_mixed_precision)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        running_train_loss = 0.0
        num_train_samples = 0

        for batch in train_loader:
            features = batch["features"].to(device).float()
            gene_labels = batch["labels"].to(device).float()
            smiles_tokens = batch["smiles_tokens"].to(device).long()
            dosages = batch["dosage"].to(device).float()
            viability_labels = batch.get("viability", None)
            if viability_labels is not None:
                viability_labels = viability_labels.to(device).float()

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_mixed_precision):
                outputs = model(features, smiles_tokens, dosages)

                # Calculate loss based on task type
                if model.task_type == "multi-task":
                    loss = criterion(outputs, gene_labels, viability_labels)
                elif model.task_type == "gene-expression":
                    loss = criterion(outputs, gene_labels, None)
                elif model.task_type == "viability":
                    loss = criterion(outputs, None, viability_labels)
                else:
                    raise ValueError(f"Unsupported task type: {model.task_type}")

            scaler.scale(loss).backward()

            if gradient_clipping is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

            scaler.step(optimizer)
            scaler.update()

            batch_size = features.size(0)
            running_train_loss += loss.item() * batch_size
            num_train_samples += batch_size

        epoch_train_loss = running_train_loss / num_train_samples
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        num_val_samples = 0

        with torch.no_grad(), autocast(enabled=use_mixed_precision):
            for batch in val_loader:
                features = batch["features"].to(device).float()
                gene_labels = batch["labels"].to(device).float()
                smiles_tokens = batch["smiles_tokens"].to(device).long()
                dosages = batch["dosage"].to(device).float()
                viability_labels = batch.get("viability", None)
                if viability_labels is not None:
                    viability_labels = viability_labels.to(device).float()

                outputs = model(features, smiles_tokens, dosages)

                # Calculate validation loss
                if model.task_type == "multi-task":
                    val_loss = criterion(outputs, gene_labels, viability_labels)
                elif model.task_type == "gene-expression":
                    val_loss = criterion(outputs, gene_labels, None)
                elif model.task_type == "viability":
                    val_loss = criterion(outputs, None, viability_labels)
                else:
                    raise ValueError(f"Unsupported task type: {model.task_type}")

                batch_size = features.size(0)
                running_val_loss += val_loss.item() * batch_size
                num_val_samples += batch_size

        epoch_val_loss = running_val_loss / num_val_samples
        val_losses.append(epoch_val_loss)

        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        logging.info(
            f"Epoch {epoch + 1}/{epochs} - {model_name}, "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logging.info(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save the model
    torch.save(model.state_dict(), f"./{model_name}.pt")

    return train_losses, val_losses
