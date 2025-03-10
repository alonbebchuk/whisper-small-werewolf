import numpy as np
import os
import time

from src.new.datasets import get_dataset
from src.new.data_collators import get_data_collator
from torch.utils.data import DataLoader


_dataloaders = None


def get_dataloaders(model_name, training_args):
    global _dataloaders
    if _dataloaders is None:
        dataset = get_dataset(model_name)
        data_collator = get_data_collator(model_name)
        common_kwargs = dict(
            collate_fn=data_collator,
            num_workers=os.cpu_count() // 2,
            prefetch_factor=4,
            worker_init_fn=lambda worker_id: np.random.seed(worker_id + int(time.time())),
            drop_last=True,
            persistent_workers=True,
            pin_memory=True,
        )
        train_loader = DataLoader(dataset["train"], batch_size=training_args.train_batch_size, shuffle=True, **common_kwargs)
        validation_loader = DataLoader(dataset["validation"], batch_size=training_args.eval_batch_size, shuffle=False, **common_kwargs)
        _dataloaders = (train_loader, validation_loader)
    return _dataloaders
