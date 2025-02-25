from dataclasses import dataclass
from transformers import WhisperProcessor
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration


BATCH_SIZE = 50
BOS_LEN = 2
EOS_LEN = 1
MAX_DURATION = 30
MASK_ID = -100
MAX_LENGTH = 448
SAMPLING_RATE = 16000
WORD_ERROR_PENALTY = 100

  

import numpy as np
from datasets import load_dataset, Audio, DatasetDict, Dataset



# batch


from dataclasses import dataclass
from transformers import WhisperProcessor

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features):
      batch = self.processor.feature_extractor.pad([{"input_features": feature} for feature in features["input_features"]], return_tensors="pt")

      
      decoder_input_ids_batch = self.processor.tokenizer.pad([{"input_ids": feature} for feature in features["decoder_input_ids"]], return_tensors="pt")
      batch["decoder_input_ids"] = decoder_input_ids_batch["input_ids"]

      labels_batch = self.processor.tokenizer.pad([{"input_ids": feature} for feature in features["labels"]], return_tensors="pt")
      labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), MASK_ID)
      batch["labels"] = labels

      return batch



prompt_prefix, prompt_suffix = """Given the following dialogue and audio, assign the last utterance one or more of the following tags (delimited by commas):
'Accusation', 'Defense', 'Evidence', 'Identity Declaration', 'Interrogation', 'No Strategy'

```
""", """
```

Reminder - Assign one or more of the following tags to the last utterance (delimited by commas):
'Accusation', 'Defense', 'Evidence', 'Identity Declaration', 'Interrogation', 'No Strategy'

Assignment:
"""

def create_prepare_decoder_input_ids_and_labels_fn(model_name):
    tokenizer = WhisperTokenizer.from_pretrained(model_name)
    prompt_prefix_len = len(tokenizer.encode(prompt_prefix)) - EOS_LEN
    max_prompt_len = MAX_LENGTH - prompt_prefix_len
    def prepare_decoder_input_ids_and_labels(sample):
        dialogue = "\n".join(f"{x['speaker']}: {x['utterance']}" for x in sample["dialogue"])
        target = sample["dialogue"][-1]["target"]
        text = prompt_prefix + dialogue + prompt_suffix + target

        input_ids = tokenizer.encode(text)
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:prompt_prefix_len] + input_ids[-max_prompt_len:]

        decoder_input_ids = np.array(input_ids)
        sample["decoder_input_ids"] = decoder_input_ids

        labels = np.array(input_ids)
        target_len = len(tokenizer.encode(target)) - BOS_LEN
        labels[:-target_len] = MASK_ID
        sample["labels"] = labels
        sample["target"] = target.split(", ")
        # labels[target_len:] = tokenizer.encode(target)

        return sample

    return prepare_decoder_input_ids_and_labels, tokenizer


def create_prepare_audio_fn(model_name):
  feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
  def prepare_audio(batch):
    audio_arrays = [x["array"] for x in batch["audio"]]
    batch["input_features"] = feature_extractor(audio_arrays, sampling_rate=SAMPLING_RATE, return_tensors="pt").input_features


    return batch

  return prepare_audio, feature_extractor

from datasets import load_dataset

def load_werewolf_data():
    # werewolf_data = DatasetDict()
    # werewolf_data["train"] = load_dataset("parquet", data_files=["https://huggingface.co/datasets/iohadrubin/werewolf_dialogue_data_10sec/resolve/main/data/train-00002-of-00014-e0e6b0000eedceb4.parquet"])["train"]
    # werewolf_data["test"] = load_dataset("parquet", data_files=["https://huggingface.co/datasets/iohadrubin/werewolf_dialogue_data_10sec/resolve/main/data/train-00009-of-00014-603088eeb6352a9b.parquet"])["train"]
    
    return load_dataset("iohadrubin/werewolf_dialogue_data_10sec")
    # return werewolf_data

def filter_data(werewolf_data):
    def filter_fn(x):
        duration_valid = x["end"] - x["start"] <= MAX_DURATION
        if not duration_valid:
            return False
        has_target = x["dialogue"][-1]["target"] is not None 
        if not has_target:
            return False
        target_not_empty = len(x["dialogue"][-1]["target"].strip()) > 0
        if not target_not_empty:
            return False
        return True
    
    werewolf_data = werewolf_data.filter(filter_fn)
    return werewolf_data


def load_and_prepare_werewolf_data(model_name):

    werewolf_data = load_werewolf_data()
    werewolf_data = filter_data(werewolf_data)

    # processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    prepare_decoder_input_ids_and_labels, tokenizer = create_prepare_decoder_input_ids_and_labels_fn(model_name)
    prepare_audio, feature_extractor = create_prepare_audio_fn(model_name)
    werewolf_data = werewolf_data.map(prepare_audio, batched=True, batch_size=BATCH_SIZE)
    werewolf_data = werewolf_data.map(prepare_decoder_input_ids_and_labels)
    werewolf_data = werewolf_data.remove_columns(["start", "end", "idx", "Game_ID", "file_name", "video_name", "startRoles", "startTime", "endRoles", "playerNames"])
    return werewolf_data, tokenizer, feature_extractor




