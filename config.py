import torch
import numpy as np
import random

def get_config():
    """Returns a dictionary containing model and training configurations."""
    seed = 1234  # Define the seed for reproducibility

    config = {
        "DEVICE": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "img_size": (28, 28, 1),  # (width, height, channels)
        "timestep_embedding_dim": 256,
        "n_layers": 8,
        "hidden_dim": 256,
        "hidden_dims": [256 for _ in range(8)],
        "n_timesteps": 1000,
        "beta_minmax": [1e-4, 2e-2],
        "train_batch_size": 128,
        "inference_batch_size": 64,
        "lr": 5e-5,
        "seed": seed,
        "epochs": 200,
        "save_epochs": [0, 100, 200],

        "paths": {
            "train_data": "/pgeoprj/godeep/ej44/gan/MNIST/mnist_train.csv", # Modify your path here
            "test_data": "/pgeoprj/godeep/ej44/gan/MNIST/mnist_test.csv", # Modify your path here
            "checkpoint_dir": "checkpoints"
        }
    }

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    return config