import os

os.environ["HF_DATASETS_CACHE"] = "/dev/shm/hf_cache"

from datasets import Audio, load_dataset
from transformers import WhisperFeatureExtractor

max_audio_len = 480000
max_dialogue_len = 10
num_proc = 10

strategies = {"Accusation", "Call for Action", "Defense", "Evidence", "Identity Declaration", "Interrogation"}

audio = Audio()
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")


def process_sample(sample):
    target = sample["dialogue"][-1]["target"]
    if target is None:
        target_strategies = None
    else:
        target_strategies = set(target.split(", ")) & strategies

    dialogue = []
    for d in sample["dialogue"][-max_dialogue_len:]:
        dialogue.extend([d["speaker"], ": ", d["utterance"], "\n"])
    dialogue = "".join(dialogue)

    if "array" in sample["audio"]:
        audio_array = sample["audio"]["array"]
    else:
        audio_array = audio.decode_example(sample["audio"])["array"]
    audio_array = audio_array[-max_audio_len:]
    input_features = feature_extractor(audio_array, sampling_rate=feature_extractor.sampling_rate).input_features[0]

    return {"target_strategies": target_strategies, "dialogue": dialogue, "input_features": input_features}


dataset = load_dataset("iohadrubin/werewolf_dialogue_data_10sec", num_proc=num_proc)
dataset = dataset.remove_columns(["start", "end", "startTime", "startRoles", "endRoles", "playerNames"])
dataset = dataset.map(process_sample, num_proc=num_proc, remove_columns=["dialogue", "audio"])
dataset = dataset.filter(lambda sample: sample["target_strategies"] is not None, num_proc=num_proc)
dataset.push_to_hub("alonbeb/werewolf-data")
