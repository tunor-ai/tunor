from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load the original GPT-2 model
original_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load the finetuned model (assumed to be saved locally)
finetuned_model = GPT2LMHeadModel.from_pretrained('/scratch/jd5018/tunor/open-instruct/output/tulu_v2_gpt2/')

def generate_text(model, tokenizer, prompt, max_length=128):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define the input prompt
prompt = "What is a potential application for AI technology?"

# Generate output from the original model
original_output = generate_text(original_model, tokenizer, prompt)

# Generate output from the finetuned model
finetuned_output = generate_text(finetuned_model, tokenizer, prompt)

# Print and compare the outputs
print("Original GPT-2 Output:\n", original_output)
print("\nFinetuned GPT-2 Output:\n", finetuned_output)
