import wandb


def init_wandb():
    wandb.init(
        project="5ARG45",
        name="perturbinator",
        mode="offline",
        config={
            "lr": 0.001,
            "architecture": "Perturbinator",
            "dataset": "LINCS/CTRPv2",
            "epochs": 30,
            "batch_size": 1024,
        },
    )

    return wandb.config
