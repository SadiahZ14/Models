# Gollum Tonality Fine-Tuned LLaMA-2 Model

<img src="https://cdn-uploads.huggingface.co/production/uploads/6564e76de6b20bc37e494589/wcj1pIDVKbhkyi_DBdAPV.png" width="600" alt="Gollum Tonality Fine-Tuned LLAMA-2 7B Model inference code">

## Overview
This model is a fine-tuned version of the LLAMA-2 7B model, specifically trained to generate responses with a tonality similar to the character Gollum from J.R.R. Tolkien's "The Lord of the Rings" series. The model has been fine-tuned using a dataset of Gollum's dialogue and text samples to capture his unique speaking style, mannerisms, and personality.

## Model Details
* **Base Model**: "NousResearch/Llama-2-7b-chat-hf"
* **Fine-Tuning Dataset**: Custom dataset of Gollum's dialogue and text samples.
* **Fine-Tuning Approach**: PEFT (LoRA) and SFT Trainer.
* **Model Size**: The model retains the same size and architecture as the original LLaMA model.

## Intended Use
The Gollum Tonality Fine-Tuned LLaMA Model is designed to generate responses and engage in conversations with a tonality and personality similar to the character Gollum. It can be used for various creative and entertainment purposes, such as:
* Generating Gollum-like dialogue for stories, fan fiction, or roleplaying scenarios
* Creating interactive chatbots or virtual assistants with Gollum's personality
* Enhancing natural language processing applications with a unique and recognizable tonality

## Limitations and Considerations
* The model's responses are generated based on patterns and characteristics learned from the fine-tuning dataset. While it aims to capture Gollum's tonality, the generated text may not always perfectly align with Gollum's canonical dialogue or behavior.
* The model may generate responses that are biased or inconsistent with Gollum's character at times, as it is still an AI language model and not a perfect replication of the original character.
* The generated text should be used responsibly and with awareness of its fictional nature. It should not be considered a substitute for professional writing or official "The Lord of the Rings" content.

## Inference Code
To test and interact with the Gollum Tonality Fine-Tuned LLaMA Model, you can use the following inference code:
```python
#Import necessary libraries
import torch
import transformers

# Load the Gollum model from hugging face
tokenizer = AutoTokenizer.from_pretrained("Sadiah/Gollum",trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Sadiah/Gollum",trust_remote_code=True,device_map= {"": 0})

# Define the input text for which you want to generate an answer
input_text = '''What is the best way to live life?[/INST]'''

# Tokenize the input text using a predefined tokenizer. 
input_ids = tokenizer(input_text, return_tensors="pt")

# Move the tokenized input to GPU memory for faster processing by specifying `.to("cuda")`.
input_ids = input_ids.to("cuda")

# Generate output sequences (answers) from the input.
outputs = model.generate(**input_ids, max_length=100, num_return_sequences=1)

# Decode the generated output back to text. `outputs[0]` accesses the first (and only, in this case) sequence.
generated_text = tokenizer.decode(outputs[0])

# Stripping and cleaning the output
answer = generated_text.split("[/INST]")[1].strip()
answer = answer.replace("</s>", "").strip()
last_full_stop_pos = answer.rfind(".")
if last_full_stop_pos != -1:
    answer = answer[:last_full_stop_pos + 1]

# Print the final, cleaned answer.
print(answer)
```

`Oh, precious, the best way to live, yes, yes. We listens to the wise, we does. First, we takes care of ourselves, yes. Then, we helps others, precious. We lives for the now, and for the future, yes. And always, always, we remembers the precious, yes. Live for the moments, and for the long, long days.`

This code snippet allows you to provide an input prompt and generate a response from the model. The generated text will aim to mimic Gollum's tonality and personality based on the fine-tuning process.

## Contact and Feedback
If you have any questions, feedback, or concerns regarding the Gollum Tonality Fine-Tuned LLaMA Model, please contact me https://www.sadiahzahoor.com/contact.
