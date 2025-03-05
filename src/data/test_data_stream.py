# python3.10 -m src.data.test_data_stream
import jax
import numpy as np
from src.config.get_config import BERT_CONFIG, get_config, WHISPER_CONFIG
from src.data.data_stream import BertDataStream, WhisperDataStream
from transformers import AutoTokenizer, AutoFeatureExtractor
from tqdm.auto import tqdm


def get_bert_config_and_stream():
    bert_config = get_config(BERT_CONFIG)
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_config.model.name, use_fast=True)
    return bert_config, BertDataStream(bert_config, bert_tokenizer)


def get_whisper_stream(use_audio, use_dialogue):
    whisper_config = get_config(WHISPER_CONFIG(use_audio, use_dialogue))
    whisper_tokenizer = AutoTokenizer.from_pretrained(whisper_config.model.name, use_fast=True)
    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_config.model.name)
    return whisper_config, WhisperDataStream(whisper_config, whisper_tokenizer, whisper_feature_extractor)


def test_stream(config, stream):
    pbar = tqdm(range(config.training.total_steps), desc="Training")
    eval_counter = config.evaluation.eval_freq
    for _, batch in zip(pbar, stream.get_iter("train")):
        print(jax.tree.map(np.shape, batch))
        print()
        eval_counter -= 1
        if eval_counter == 0:
            print("Training Batch:")
            print(batch.keys())
            eval_counter = config.evaluation.eval_freq
            for i, dev_batch in enumerate(stream.get_iter("validation")):
                if i >= config.evaluation.eval_steps:
                    print("Validation Batch:")
                    print(dev_batch.keys())
                    break
                dev_batch.pop("epoch", 0)


print("Started Testing Bert")
test_stream(*get_bert_config_and_stream())
print("Finished Testing Bert")
print("Started Testing Whisper Audio+Dialogue")
test_stream(*get_whisper_stream(True, True))
print("Finsidhed Testing Whisper Audio+Dialogue")
print("Started Testing Whisper Audio")
test_stream(*get_whisper_stream(True, False))
print("Finsidhed Testing Whisper Audio")
print("Started Testing Whisper Dialogue")
test_stream(*get_whisper_stream(False, True))
print("Finsidhed Testing Whisper Dialogue")
