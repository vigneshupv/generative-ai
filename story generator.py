from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model_and_tokenizer(model_name="gpt2-medium"):
    """Load pre-trained GPT-2 model and tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """Generate text given a prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, temperature=temperature, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    model, tokenizer = load_model_and_tokenizer()

    prompt = "Once upon a time"
    max_length = 1000
    temperature = 0.7

    generated_story = generate_text(model, tokenizer, prompt, max_length, temperature)

    print(generated_story)

if __name__ == "__main__":
    main()
