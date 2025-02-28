from datasets import load_dataset
from transformers import WhisperTokenizer


strategies = ["Accusation", "Defense", "Evidence", "Identity Declaration", "Interrogation", "No Strategy", "Call for Action"]
prompt_format = """Given the previous audio and the following dialogue, determine whether the last utterance in the following spoken dialogue fits under the strategy category of: {strategy}.
Respond with a single word: Yes or No.
Dialogue:
```
{dialogue}
```
Does the last utterance fit the strategy category {strategy}?
Completion:
"""


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


def create_into_prompt_completion_fn(strategy, model_name="openai/whisper-small", bos_token="<|startoftranscript|>", max_length=447):
    tokenizer = WhisperTokenizer.from_pretrained(model_name, bos_token=bos_token)

    def into_prompt_completion(sample):
        i = 0
        prompt = None
        completion = "Yes" if strategy in sample["dialogue"][-1]["target"].split(", ") else "No"
        while i < len(sample["dialogue"]):
            curr_dialogue = sample["dialogue"][i:]
            dialogue = "\n".join(f"{x['speaker']}: {x['utterance']}" for x in curr_dialogue)
            prompt = prompt_format.format(strategy=strategy, dialogue=dialogue)
            input_ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
            if len(input_ids) <= max_length:
                break
            i += 1
        return {"prompt": prompt, "strategy": strategy, "completion": completion}

    return into_prompt_completion


werewolf_data = load_werewolf_data()
werewolf_data = filter_data(werewolf_data)
for strategy in strategies:
    into_prompt_completion = create_into_prompt_completion_fn(strategy)
    strategy_werewolf_data = werewolf_data.map(into_prompt_completion, num_proc=50)
    strategy_werewolf_data = strategy_werewolf_data.filter(lambda sample: sample["prompt"] is not None)
    strategy_werewolf_data = strategy_werewolf_data.shuffle()
    strategy_werewolf_data.push_to_hub(f"alonbeb/werewolf_{strategy.replace(' ', '-')}_data")
