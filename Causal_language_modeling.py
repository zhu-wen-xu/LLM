from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import math


def preprocess_function(examples):
    results = tokenizer([" ".join(x) for x in examples["answers.text"]])
    return results


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


if __name__ == '__main__':
    eli5 = load_dataset("eli5_category", split="train[:5000]", trust_remote_code=True)
    eli5 = eli5.train_test_split(test_size=0.2)

    print(eli5["train"][0])

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    eli5_fl = eli5.flatten()
    print(eli5_fl["train"][0])

    tokenized_eli5 = eli5_fl.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=eli5_fl["train"].column_names,
    )

    block_size = 128

    lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=1)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

    training_args = TrainingArguments(
        output_dir="my_awesome_eli5_clm-model",
        eval_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        # push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    print(eli5)
