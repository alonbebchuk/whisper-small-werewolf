python3.10 -m src.bert.run_qa \
  --model_name_or_path google-bert/bert-base-uncased \
  --do_train   \
  --do_eval   \
  --eval_steps 1000 \
  --output_dir ./bert-werewolf #\
  # --push_to_hub \
  # --hub_model_name bert-werewolf