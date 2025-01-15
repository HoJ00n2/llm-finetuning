def training_function(script_args, training_args):
    # 데이터셋 불러오기
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, "train_dataset.json"),
        split="train",
    )
    test_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, "test_dataset.json"),
        split="train",
    )

    # 토크나이저 및 데이터셋 chat_template으로 변경
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token # 맨끝에 붙일 eos token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    tokenizer.padding_side = 'right'
    
    def template_dataset(examples):
        return{"text": tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])

    # 데이터가 chat template으로 변화되었는지 확인하기 위해 2개만 출력
    with training_args.main_process_first(
        desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(train_dataset)), 2): # train_datset중 2개 랜덤으로 뽑음
            print(train_dataset[index]["text"]) # 형식 출력 text -> 대화형 prompt
    
    # Model 및 파라미터 설정하기
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        attn_implementation="sdpa", # flash-attn, lightning-attn ..
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    # 근데 training_args는 어디서 온거지?
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Train 설정
    trainer = SFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        eval_dataset=test_dataset,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
