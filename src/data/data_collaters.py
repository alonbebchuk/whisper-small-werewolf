import numpy as np
import random

from src.data.process_dataset import strategies

strategies_len = len(strategies)


class BertDataCollator:
    def __call__(self, features):
        strategy_list = []
        label_list = []
        input_ids_list = []
        attention_mask_list = []
        for feature in features:
            bert_choice = feature["bert_choices"][random.randrange(strategies_len)]
            strategy_list.append(bert_choice["strategy"])
            label_list.append(bert_choice["label"])
            input_ids_list.append(np.array(bert_choice["input_ids"], dtype=np.int32))
            attention_mask_list.append(np.array(bert_choice["attention_mask"], dtype=np.int32))

        batch = {
            "strategy": np.array(strategy_list, dtype=object),
            "label": np.array(label_list, dtype=np.int32),
            "input_ids": np.array(input_ids_list, dtype=np.int32),
            "attention_mask": np.array(attention_mask_list, dtype=np.int32),
        }
        return batch


class WhisperDataCollator:
    def __call__(self, features):
        strategy_list = []
        label_list = []
        input_features_list = []
        decoder_input_ids_list = []
        target_tokens_list = []
        attention_mask_list = []
        loss_mask_list = []
        for feature in features:
            whisper_choice = feature["whisper_choices"][random.randrange(strategies_len)]
            strategy_list.append(whisper_choice["strategy"])
            label_list.append(whisper_choice["label"])
            input_features_list.append(np.array(feature["input_features"], np.float32))
            decoder_input_ids_list.append(np.array(whisper_choice["decoder_input_ids"], dtype=np.int32))
            target_tokens_list.append(np.array(whisper_choice["target_tokens"], dtype=np.int32))
            attention_mask_list.append(np.array(whisper_choice["attention_mask"], dtype=np.int32))
            loss_mask_list.append(np.array(whisper_choice["loss_mask"], dtype=np.float32))

        batch = {
            "strategy": np.array(strategy_list, dtype=object),
            "label": np.array(label_list, dtype=np.int32),
            "input_features": np.array(input_features_list, np.float32),
            "decoder_input_ids": np.array(decoder_input_ids_list, dtype=np.int32),
            "target_tokens": np.array(target_tokens_list, dtype=np.int32),
            "attention_mask": np.array(attention_mask_list, dtype=np.int32),
            "loss_mask": np.array(loss_mask_list, dtype=np.float32),
        }
        return batch
