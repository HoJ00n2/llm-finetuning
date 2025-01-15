result = []
for index in tqdm(range(0, 1000)):
    messages = test_dataset[index]["messages"][:2]

    terminators = [
        tokenizer.eos_token_id
    ]

    # Test on sample
    input_ids = tokenizer.apply_chat_template(
        messages, add_generate_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

    response = outputs[0][input_ids.shape[-1]:]
    question = test_dataset[index]["messages"][1]['content']
    answer = test_dataset[index]["messages"][2]['content']
    generation = tokenizer.decode(response, skip_special_tokens=True)
    result.append([question, answer, generation]) # [질문, 정답, 예측] list를 result에 저장

with open("./test/model_generation_result.txt", "w") as file:
    for line in result:
        file.write(str(line) + "\n")