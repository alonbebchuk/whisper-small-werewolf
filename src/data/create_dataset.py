from datasets import load_dataset
from src.common.config import get_config, get_strategy_dataset_name
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


def filter_data(config, dataset):
    def filter_fn(x):
        duration = x["end"] - x["start"]
        if duration > config.model.max_duration:
            return False

        target = x["dialogue"][-1]["target"]
        if target is None or len(target.strip()) == 0:
            return False

        return True

    dataset = dataset.filter(filter_fn)
    return dataset


def create_into_prompt_completion_fn(config, strategy):
    tokenizer = WhisperTokenizer.from_pretrained(config.model.name, bos_token=config.model.bos_token)

    def into_prompt_completion(sample):
        completion = "Yes" if strategy in sample["dialogue"][-1]["target"].split(", ") else "No"

        prompt = None
        for i in range(len(sample["dialogue"])):
            curr_dialogue = "\n".join(f"{x['speaker']}: {x['utterance']}" for x in sample["dialogue"][i:])
            curr_prompt = prompt_format.format(strategy=strategy, dialogue=curr_dialogue)

            input_ids = tokenizer.encode(curr_prompt + completion, add_special_tokens=False)
            if len(input_ids) < config.model.max_len:
                prompt = curr_prompt
                break

        return {"prompt": prompt, "strategy": strategy, "completion": completion}

    return into_prompt_completion


def main():
    config = get_config()

    base_dataset = load_dataset(config.dataset.base_name)
    base_dataset = filter_data(config, base_dataset)

    for strategy in strategies:
        into_prompt_completion = create_into_prompt_completion_fn(config, strategy)
        strategy_data = base_dataset.map(into_prompt_completion, num_proc=50)
        strategy_data = strategy_data.shuffle()
        strategy_data = strategy_data.filter(lambda sample: sample["prompt"] is not None)

        strategy_dataset_name = get_strategy_dataset_name(config, strategy)
        strategy_data.push_to_hub(strategy_dataset_name)


if __name__ == "__main__":
    main()
