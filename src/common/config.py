import yaml

from ml_collections.config_dict import ConfigDict


CONFIG = """
model:
  name: "openai/whisper-small"
  name_format: "whisper_small_werewolf_{model_type}"
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
  total_steps: 1000
  warmup_steps: 100
  lr: 5e-5
  wd: 0.01
  b2: 0.95
  batch_size: 64
evaluation:
  eval_freq: 20
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


def get_model_name(config, use_audio, use_dialogue):
    model_name_format = config.model.name_format

    if not use_audio:
        model_type = "dialogue_only"
    elif not use_dialogue:
        model_type = "audio_only"
    else:
        model_type = "audio_and_dialogue"

    model_name = model_name_format.format(model_type=model_type)
    return model_name


strategies = ["Accusation", "Defense", "Evidence", "Identity Declaration", "Interrogation", "No Strategy", "Call for Action"]
prompt_with_dialogue_format = """Given the previous audio and the following dialogue, determine whether the last utterance in the following spoken dialogue fits under the strategy category of: {strategy}.
Respond with a single word: Yes or No.
Dialogue:
```
{dialogue}
```
Does the last utterance fit the strategy category {strategy}?
Completion:
"""
prompt_without_dialogue_format = """Given the previous audio, determine whether the last utterance fits under the strategy category of: {strategy}.
Respond with a single word: Yes or No.
Does the last utterance fit the strategy category {strategy}?
Completion:
"""