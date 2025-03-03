python3.10 -m src.bert.run_qa \
  --model_name_or_path google-bert/bert-base-uncased \
  --output_dir ./bert-werewolf \
  --eval_steps 1000 \
  --do_train   \
  --do_eval   \
  --push_to_hub