import wandb

def init_wandb():
    wandb.init(
        project="5ARG45",
        name="perturbinator",
        mode="online",
    )

    wandb.config = {
        "lr": 0.001,
        "architecture": "Perturbinator",
        "dataset": "LINCS/CTRPv2",
        "epochs": 20,
        "batch_size": 1024,
    }
    return wandb.config