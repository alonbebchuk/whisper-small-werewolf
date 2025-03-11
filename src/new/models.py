from transformers import FlaxBertForSequenceClassification, FlaxWhisperForConditionalGeneration

_model = None


def get_model(model_name):
    global _model
    if _model is None:
        if model_name == "bert":
            _model = FlaxBertForSequenceClassification.from_pretrained("google-bert/bert-base-cased")
        elif model_name == "whisper":
            _model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        else:
            raise Exception(f"Model name {model_name} is not supported.")
    return _model
