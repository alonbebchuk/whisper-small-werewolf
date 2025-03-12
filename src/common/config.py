import yaml

from ml_collections.config_dict import ConfigDict


BERT_CONFIG = """
evaluation:
  batch_size: 64
  eval_freq: 20
  eval_steps: 5
metrics:
  rolling_average_window: 10
model:
  name: "google-bert/bert-base-cased"
training:
  b2: 0.99
  batch_size: 64
  lr: 5e-6
  total_steps: 1000
  warmup_steps: 100
  wd: 0.001
"""


WHISPER_CONFIG = f"""
evaluation:
  batch_size: 64
  eval_freq: 20
  eval_steps: 5
metrics:
  rolling_average_window: 10
model:
  name: "openai/whisper-small"
training:
  b2: 0.95
  batch_size: 64
  lr: 5e-5
  total_steps: 1000
  warmup_steps: 100
  wd: 0.01
"""


def get_config(model_name):
    if "bert" in model_name:
<<<<<<< HEAD
      return ConfigDict(yaml.safe_load(BERT_CONFIG))
    elif "whisper" in model_name:
      return ConfigDict(yaml.safe_load(WHISPER_CONFIG))
=======
        return ConfigDict(yaml.safe_load(BERT_CONFIG))
    elif "whisper" in model_name:
        return ConfigDict(yaml.safe_load(WHISPER_CONFIG))
>>>>>>> f47d584573fc5700b20b2941189b34e5c055ec01
    else:
        raise Exception(f"Model name {model_name} is not supported.")
