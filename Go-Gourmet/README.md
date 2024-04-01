---
library_name: transformers
license: cc-by-nc-4.0
datasets: Sadiah/Go-Gourmet
language: en
---

# Go-Gourmet Fine-Tuned Mistral 7B Model

<img src="https://github.com/SadiahZ14/Models/assets/100665526/43a11956-ba88-483e-afa7-653448697070" width="600" alt="Go-Gourmet">

## Overview
The Go-Gourmet model is a fine-tuned version of the `mistralai/Mistral-7B-Instruct-v0.2` base model with 32k context window, specifically trained to generate structured restaurant cards based on an input of a restaurant name and location. The model has been fine-tuned using a custom dataset of restaurant information to capture relevant details such as cuisine, opening times, location, rating, average price, best dishes, pre-booking requirements, dress code, and website.

## Model Details
* **Base Model**:  `mistralai/Mistral-7B-Instruct-v0.2`.
* **Fine-Tuning Dataset**: Custom dataset of restaurant information `Sadiah/Go-Gourmet`.
* **Fine-Tuning Approach**: `QLoRA` and `SFT Trainer`.
* **Model Size**: The model retains the same size and architecture as the original Mistral base model.

## Intended Use
The Go-Gourmet Fine-Tuned Mistral Model is designed to generate structured restaurant cards based on an input of a restaurant name and location. It can be used for various purposes, such as:
* Generating informative restaurant cards for food and travel applications
* Providing quick and structured information about restaurants to users
* Enhancing natural language processing applications related to the food and hospitality industry

## Limitations and Considerations
* The model's outputs are generated based on patterns and characteristics learned from the fine-tuning dataset. While it aims to provide accurate and relevant information, the generated restaurant cards may not always be perfect or up-to-date.
* The model relies on the quality and comprehensiveness of the fine-tuning dataset. If certain details or categories are missing from the dataset, the model may not be able to generate them accurately.
* The generated restaurant cards should be used as a starting point and should be verified with official sources or the restaurants themselves for the most accurate and current information.

## Inference Code
To test and interact with the Go-Gourmet Fine-Tuned Mistral Model, you can use the following inference code:

```python
# Load the fine-tuned model from hugging face
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer_new = AutoTokenizer.from_pretrained("Sadiah/Go-Gourmet")
model_new = AutoModelForCausalLM.from_pretrained(
    "Sadiah/Go-Gourmet", 
    trust_remote_code=True, 
    torch_dtype=torch.float16,
    device_map= {"": 0},
)

input_text = '''[INST]Olives, Delhi [/INST]''' #Define instruction
input_ids = tokenizer_new(input_text, return_tensors="pt") #Tokenize instruction
input_ids = input_ids.to("cuda") #Move instruction to GPU
outputs = model_new.generate(**input_ids, max_length=300, num_return_sequences=1, temperature=0) #Generate response
generated_text = tokenizer_new.decode(outputs[0]) #Decode generated response

# Find the index of the closing instruction tag and remove the instruction
instruction_end_index = generated_text.find("[/INST]")
if instruction_end_index != -1:
    generated_text = generated_text[instruction_end_index + len("[/INST]"):].strip()

# Find the index of "Website:" and the end of the website address
website_start_index = generated_text.find("Website:")
if website_start_index != -1:
    website_end_index = generated_text.find("\n", website_start_index)
    if website_end_index == -1:
        website_end_index = len(generated_text)
    truncated_text = generated_text[:website_end_index]
    print(truncated_text.strip())
else:
    print(generated_text)
```

```
Name: Olives, Delhi 
Cuisine: Mediterranean 
Opening Times: Mon-Sun: 12:30pm-3:30pm, 7pm-11:30pm 
Location: 1, Kalka Das Marg, New Delhi, Delhi 110001, India (28.7031, 77.1123) 
Rating: 4.3 (Source: Zomato) 
Average Price Per Person: Moderate 
Three Best Dishes: 
1. Grilled Halloumi Cheese: A popular Mediterranean cheese, grilled to perfection.
2. Falafel Platter: A selection of crispy, flavorful falafel balls served with hummus and pita bread.
3. Lamb Shank Tagine: Slow-cooked lamb shank in a rich, aromatic sauce.
Pre-Booking Needed: Recommended, especially for dinner 
Dress Code: Casual 
Website: http://www.olivesdelhi.com/
```

This code snippet allows you to provide an input of a restaurant name and location and generate a structured restaurant card using the Go-Gourmet Fine-Tuned Mistral Model.

## Contact and Feedback
If you have any questions, feedback, or concerns regarding the Go-Gourmet Fine-Tuned Mistral Model, please contact me at https://www.sadiahzahoor.com/contact .
