---
library_name: transformers
license: cc-by-nc-4.0
datasets: Sadiah/Genie
language: en
---
# Genie Tonality Fine-Tuned Mistral 7B Model

<img src="https://cdn-uploads.huggingface.co/production/uploads/6564e76de6b20bc37e494589/_7MRpFY2lpc4aGQPybqU_.png" width="600" alt="Genie Tonality Fine-Tuned Mixtral-8x7B Model overview">

[Hugging Face Model - Genie](https://huggingface.co/Sadiah/Genie)

## Overview

This model is a fine-tuned version of the "mistralai/Mistral-7B-v0.1" model, specifically trained to generate responses with a tonality similar to the character Genie from Disney's "Aladdin" franchise. The model has been fine-tuned using a dataset of Genie's dialogue and text samples to capture his unique speaking style, mannerisms, and personality.

## Model Details

- **Base Model**: `mistralai/Mistral-7B-v0.1`
- **Fine-Tuning Dataset**: Custom dataset of Genie's dialogue and text samples.
- **Fine-Tuning Approach**: PEFT `LoRA` and SFT Trainer.
- **Model Size**: The model retains the same size and architecture as the original Mistral 7B model.

## Intended Use

The Genie Tonality Fine-Tuned Mistral 7B Model is designed to generate responses and engage in conversations with a tonality and personality similar to the character Genie. It can be used for various creative and entertainment purposes, such as:

- Generating Genie-like dialogue for stories, fan fiction, or roleplaying scenarios
- Creating interactive chatbots or virtual assistants with Genie's personality
- Enhancing natural language processing applications with a unique and recognizable tonality

## Limitations and Considerations

- The model's responses are generated based on patterns and characteristics learned from the fine-tuning dataset. While it aims to capture Genie's tonality, the generated text may not always perfectly align with Genie's canonical dialogue or behavior.
- The model may generate responses that are biased or inconsistent with Genie's character at times, as it is still an AI language model and not a perfect replication of the original character.
- The generated text should be used responsibly and with awareness of its fictional nature. It should not be considered a substitute for professional writing or official Disney content.

## Inference Code

To test and interact with the Genie Tonality Fine-Tuned Mistral 7B Model, you can use the following inference code:

```python
# Import necessary libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the Genie model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("Sadiah/Genie")
model = AutoModelForCausalLM.from_pretrained("Sadiah/Genie", device_map={"": 0})

# Define the input text for which you want to generate an answer
input_text = "<s>[INST]What is a function? [/INST]"

# Tokenize the input text using the loaded tokenizer
input_ids = tokenizer(input_text, return_tensors="pt")

# Move the tokenized input to GPU memory for faster processing
input_ids = input_ids.to("cuda")

# Generate output sequences (answers) from the input
outputs = model.generate(**input_ids, max_length=200, num_return_sequences=1, temperature=0.7)

# Decode the generated output back to text
generated_text = tokenizer.decode(outputs[0])

# Extract the answer by removing the surrounding tags and the question
answer = generated_text.split("[/INST]")[1].strip()
answer = answer.replace("</s>", "").strip()

# Find the position of the last full stop (period)
last_full_stop_pos = answer.rfind(".")

# Extract the answer up to the last full stop
if last_full_stop_pos != -1:
    answer = answer[:last_full_stop_pos + 1]

# Print the final, cleaned answer
print(answer)
```
`Ah, Master, a function is like a magic trick! It takes an input and performs a special task, transforming it into an output. It's like a wizard's spell, turning one thing into another.`

This code snippet allows you to provide an input prompt and generate a response from the model. The generated text will aim to mimic Genie's tonality and personality based on the fine-tuning process.

## Contact and Feedback
If you have any questions, feedback, or concerns regarding the Genie's Tonality Fine-Tuned Mistral 7B Model, please contact me https://www.sadiahzahoor.com/contact.
