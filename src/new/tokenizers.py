from transformers import BertTokenizer, WhisperTokenizer

_tokenizer = None


def get_tokenizer(model_name):
    global _tokenizer
    if _tokenizer is None:
        if model_name == "bert":
            _tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", truncation_side="left")
        elif model_name == "whisper":
            _tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", truncation_side="left")
        else:
            raise Exception(f"Model name {model_name} is not supported.")
    return _tokenizer
