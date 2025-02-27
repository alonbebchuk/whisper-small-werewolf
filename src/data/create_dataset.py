from datasets import load_dataset
from transformers import WhisperTokenizer


def load_werewolf_data(dataset="iohadrubin/werewolf_dialogue_data_10sec"):
    werewolf_data = load_dataset(dataset)
    return werewolf_data


def filter_data(werewolf_data, max_duration=30):
    def filter_fn(x):
        duration = x["end"] - x["start"]
        if duration > max_duration:
            return False
        target = x["dialogue"][-1]["target"]
        if target is None or len(target.strip()) == 0:
            return False
        return True

    werewolf_data = werewolf_data.filter(filter_fn)
    return werewolf_data


prompt_prefix = """Given the following dialogue and audio, assign the last utterance one or more of the following tags (delimited by commas):
'Accusation', 'Defense', 'Evidence', 'Identity Declaration', 'Interrogation', 'No Strategy'

```
"""
prompt_suffix = """
```
Reminder - Assign one or more of the following tags to the last utterance (delimited by commas):
'Accusation', 'Defense', 'Evidence', 'Identity Declaration', 'Interrogation', 'No Strategy'
Assignment:
"""


def create_into_prompt_completion_fn(model_name="openai/whisper-small", bos_token="<|startoftranscript|>", max_length=447):
    tokenizer = WhisperTokenizer.from_pretrained(model_name, bos_token=bos_token)

    def into_prompt_completion(sample):
        i = 0
        while i < len(sample["dialogue"]):
            curr_dialogue = sample["dialogue"][i:]
            dialogue = "\n".join(f"{x['speaker']}: {x['utterance']}" for x in curr_dialogue)
            prompt = prompt_prefix + dialogue + prompt_suffix
            target = sample["dialogue"][-1]["target"]
            input_ids = tokenizer.encode(prompt + target, add_special_tokens=False)
            if len(input_ids) <= max_length:
                return {"prompt": prompt, "completion": target}
            i += 1
        return {"prompt": None, "completion": None}

    return into_prompt_completion


into_prompt_completion = create_into_prompt_completion_fn()
werewolf_data = load_werewolf_data()
werewolf_data = filter_data(werewolf_data)
werewolf_data = werewolf_data.map(into_prompt_completion, num_proc=50)
werewolf_data = werewolf_data.filter(lambda x: x["prompt"] is not None)
werewolf_data.push_to_hub("alonbeb/werewolf_dataset")