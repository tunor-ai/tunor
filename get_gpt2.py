#TODO: REDOWNLOAD THE MODEL EVERYTIME AFTER UPDATING TRANSFORMERS

from transformers import GPT2Model, GPT2Tokenizer

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load model
model = GPT2Model.from_pretrained('gpt2')

# Now you can use `model` for inference or continue with fine-tuning
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Define the directory
save_directory = "./gpt2"

# Save tokenizer and model
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
