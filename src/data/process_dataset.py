import os

os.environ["HF_DATASETS_CACHE"] = "/dev/shm/hf_cache"

import numpy as np

from datasets import Audio, load_dataset
from transformers import WhisperFeatureExtractor
from transformers import BertTokenizer, WhisperTokenizer

completions = ["No", "Yes"]
strategies = ["Accusation", "Call for Action", "Defense", "Evidence", "Identity Declaration", "Interrogation"]


def filter_sample(sample):
    return sample["dialogue"][-1]["target"] is not None


def create_process_sample():
    max_audio_len = 480000
    max_dialogue_lookback = 10
    max_tokens_len = 448
    sampling_rate = 16000

    prompt_format = "{dialogue}Does the final utterance conform to the strategy: '{strategy}'?\nAnswer with a single word: Yes or No.\n"

    audio = Audio(sampling_rate=sampling_rate)
    bert_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", truncation_side="left")
    whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    whisper_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", truncation_side="left")

    def process_sample(sample):
        if "array" not in sample["audio"]:
            sample["audio"] = audio.decode_example(sample["audio"])
        audio_array = sample["audio"]["array"][-max_audio_len:]
        sample["input_features"] = whisper_feature_extractor(audio_array, sampling_rate=sampling_rate).input_features[0]

        target = sample["dialogue"][-1]["target"]
        target_strategies = set(target.split(", ")) if target is not None else None

        dialogue = "".join(f"{d['speaker']}: {d['utterance']}\n" for d in sample["dialogue"][-max_dialogue_lookback:])

        bert_choices = []
        whisper_choices = []
        for strategy in strategies:
            label = int(strategy in target_strategies)
            prompt = prompt_format.format(dialogue=dialogue, strategy=strategy)

            bert_result = bert_tokenizer(prompt, padding="max_length", truncation=True, max_length=max_tokens_len)
            bert_choices.append(
                {
                    "strategy": strategy,
                    "label": label,
                    "input_ids": bert_result.input_ids,
                    "attention_mask": bert_result.attention_mask,
                }
            )

            whisper_result = whisper_tokenizer([prompt + completions[label]], padding="max_length", truncation=True, max_length=max_tokens_len + 1, return_length=True)
            decoder_input_ids = whisper_result.input_ids[0][:-1]
            target_tokens = whisper_result.input_ids[0][1:]
            attention_mask = whisper_result.attention_mask[0][:-1]
            loss_mask = np.zeros(max_tokens_len, dtype=np.float32)
            loss_mask[[whisper_result.length[0] - 3, whisper_result.length[0] - 2]] = 1
            whisper_choices.append(
                {
                    "strategy": strategy,
                    "label": label,
                    "decoder_input_ids": decoder_input_ids,
                    "target_tokens": target_tokens,
                    "attention_mask": attention_mask,
                    "loss_mask": loss_mask,
                }
            )

        return {"bert_choices": bert_choices, "whisper_choices": whisper_choices}

    return process_sample


if __name__ == "__main__":
    num_proc = os.cpu_count() // 2

    dataset = load_dataset("iohadrubin/werewolf_dialogue_data_10sec", num_proc=num_proc)
    dataset = dataset.filter(filter_sample, num_proc=num_proc)

    process_sample = create_process_sample()
    dataset = dataset.map(process_sample, num_proc=num_proc)

    dataset = dataset.shuffle(seed=42)
    dataset = dataset.flatten_indices(num_proc=num_proc)
    print(dataset)
    dataset.save_to_disk("/dev/shm/hf_cache/werewolf_data", num_proc=num_proc)