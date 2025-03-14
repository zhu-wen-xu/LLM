from transformers import pipeline
import torch

pipeline = pipeline(task="text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")
prompt = """Classify the text into neutral, negative or positive.
Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
Sentiment:
"""

outputs = pipeline(prompt, max_new_tokens=10)
for output in outputs:
    print(f"Result: {output['generated_text']}")


print(outputs)