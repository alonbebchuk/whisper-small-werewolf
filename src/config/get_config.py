import yaml

from ml_collections.config_dict import ConfigDict


BERT_CONFIG = """
dataset:
  name: "iohadrubin/werewolf_dialogue_data_10sec"
  prompt_prefix_format: |
    You are an expert in dialogue strategy analysis. Review the conversation below and determine if the final utterance aligns with the specific strategy: "{strategy}".
    Consider the full context of the dialogue as you assess the intent behind the final statement.
    Dialogue:
  prompt_suffix_format: |
    Based on your analysis, does the final utterance conform to the strategy "{strategy}"?
    Answer with a single word: Yes or No.
  strategies: !!set {"Accusation", "Defense", "Evidence", "Identity Declaration", "Interrogation", "No Strategy", "Call for Action"}
  use_dialogue: True
evaluation:
  batch_size: 64
  eval_freq: 20
  eval_steps: 5
metrics:
  rolling_average_window: 10
model:
  max_seq_len: 448
  name: "google-bert/bert-base-cased"
training:
  b2: 0.95
  batch_size: 64
  lr: 5e-5
  total_steps: 1000
  warmup_steps: 100
  wd: 0.01
"""


def WHISPER_CONFIG(use_audio, use_dialogue):
    if not use_audio:
        prompt_prefix_format = """You are an expert in dialogue strategy analysis. You are provided with a transcript of a conversation. Carefully review the transcript to capture all nuances and context. Determine whether the final utterance in the conversation aligns with the specific strategy: "{strategy}".
    Dialogue:"""
        prompt_suffix_format = """Based on your analysis of the transcript, does the final utterance conform to the strategy "{strategy}"?
    Answer with a single word: Yes or No."""
    elif not use_dialogue:
        prompt_prefix_format = """You are an expert in dialogue strategy analysis. You are provided with an audio recording of a conversation. Carefully listen to the audio to capture all nuances and context. Determine whether the final utterance in the conversation aligns with the specific strategy: "{strategy}"."""
        prompt_suffix_format = """Based on your analysis of the audio, does the final utterance conform to the strategy "{strategy}"?
    Answer with a single word: Yes or No."""
    else:
        prompt_prefix_format = """You are an expert in dialogue strategy analysis. The dialogue below is a transcript derived from an audio conversation, and the original audio is also available for context. Carefully review the transcript and listen to the audio to capture all nuances and context. Determine whether the final utterance aligns with the specific strategy: "{strategy}".
    Dialogue:"""
        prompt_suffix_format = """Based on your analysis of the transcript and audio, does the final utterance conform to the strategy "{strategy}"?
    Answer with a single word: Yes or No."""
    return f"""
dataset:
  name: "iohadrubin/werewolf_dialogue_data_10sec"
  prompt_prefix_format: |
    {prompt_prefix_format}
  prompt_suffix_format: |
    {prompt_suffix_format}
  strategies: !!set {{"Accusation", "Defense", "Evidence", "Identity Declaration", "Interrogation", "No Strategy", "Call for Action"}}
  use_audio: {use_audio}
  use_dialogue: {use_dialogue}
evaluation:
  batch_size: 64
  eval_freq: 20
  eval_steps: 5
metrics:
  rolling_average_window: 10
model:
  max_audio_array_len: 480000
  max_seq_len: 448
  name: "openai/whisper-small"
  sampling_rate: 16000
training:
  b2: 0.95
  batch_size: 64
  lr: 5e-5
  total_steps: 1000
  warmup_steps: 100
  wd: 0.01
"""


def get_config(config):
    return ConfigDict(yaml.safe_load(config))
