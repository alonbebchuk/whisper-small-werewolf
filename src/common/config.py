import yaml

from ml_collections.config_dict import ConfigDict

CONFIG = """
model:
  name: "openai/whisper-small"
  bos_token: "<|startoftranscript|>"
  max_len: 448
  max_duration: 30
  sampling_rate: 16000
dataset:
  base_name: "iohadrubin/werewolf_dialogue_data_10sec"
  name_format: "alonbeb/werewolf_{strategy}_data"
figure_size:
  width: 10
  height: 5
metrics:
  rolling_average_window: 10
training:
  total_steps: 100000
  warmup_steps: 10000
  lr: 5e-5
  wd: 0.01
  b2: 0.95
  batch_size: 64
evaluation:
  eval_freq: 2000
  eval_steps: 5
  batch_size: 64
"""


def get_config():
    config_dict_raw = yaml.safe_load(CONFIG)

    config = ConfigDict(config_dict_raw)
    return config


def get_strategy_dataset_name(config, strategy):
    dataset_name_format = config.dataset.name_format

    strategy_dataset_name = dataset_name_format.format(strategy=strategy.replace(" ", "-"))
    return strategy_dataset_name
