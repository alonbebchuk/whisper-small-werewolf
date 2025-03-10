import datasets
import evaluate
import flax
import jax
import logging
import multiprocessing as mp
import os
import sys
import time
import transformers

from flax.metrics.tensorboard import SummaryWriter
from flax.training.common_utils import shard
from pathlib import Path
from src.new.dataloaders import get_dataloaders
from src.new.datasets import get_dataset, strategies, pad_token_id
from src.new.learning_rate import get_learning_rate_fn
from src.new.models import get_model
from src.new.tokenizers import get_tokenizer
from new.train_state import get_train_state
from tqdm import tqdm
from transformers import HfArgumentParser, TrainingArguments

logger = logging.getLogger(__name__)


@flax.struct.dataclass
class ModelArguments:
    model_name: str


if __name__ == "__main__":
    os.environ["HF_DATASETS_CACHE"] = "/dev/shm/hf_cache"
    mp.set_start_method("spawn", force=True)

    # 1. Parse
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    model_name = model_args.model_name

    # 2. Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
        summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    logger.info("Training parameters %s", training_args)

    # 3. Dataloaders
    training_args.device_count = jax.device_count()
    training_args.train_batch_size = training_args.per_device_train_batch_size * training_args.device_count
    training_args.eval_batch_size = training_args.per_device_eval_batch_size * training_args.device_count
    train_loader, validation_loader = get_dataloaders(model_name, training_args)

    # 4. Train State
    training_args.num_train_steps = (len(train_loader.dataset) // training_args.train_batch_size) * training_args.num_epochs
    state = get_train_state(model_name, training_args)
    state = state.replicate()

    # 5. Metrics


# python3.10 -m src.data.test --model_name="bert"
# python3.10 -m src.data.test --model_name="whisper"
