import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define paths
MODEL_PATH = "/data/coleloughbc/LLAMA-3-2/HF_Converted_llama-3-3_70B_instruct_HF"

def check_tokenizer():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("\n[Tokenizer Configuration]")
    print(f"Tokenizer class: {type(tokenizer)}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print(f"Padding side: {tokenizer.padding_side}")
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"BOS token ID: {tokenizer.bos_token_id}")

    # Tokenizing a long input to confirm actual behavior
    test_text = "This is a test sentence. " * 500  # Create a long sequence
    tokens = tokenizer(test_text, return_tensors="pt")

    print(f"Tokenized sequence length: {tokens['input_ids'].shape[1]}")
    if tokens["input_ids"].shape[1] > tokenizer.model_max_length:
        print("ERROR: Tokenized sequence exceeds model_max_length!")

    return tokenizer

def check_model():
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("\n[Model Configuration]")
    print(f"Model class: {type(model)}")
    print(f"Max position embeddings: {model.config.max_position_embeddings}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Number of attention heads: {model.config.num_attention_heads}")
    print(f"Number of hidden layers: {model.config.num_hidden_layers}")
    print(f"Intermediate size: {model.config.intermediate_size}")
    print(f"Rope scaling: {model.config.rope_scaling}")
    print(f"Rope theta: {model.config.rope_theta}")
    print(f"Vocabulary size: {model.config.vocab_size}")
    print(f"Model dtype: {model.config.torch_dtype}")

    # Verify model max position embeddings
    if model.config.max_position_embeddings < 8192:
        print("WARNING: Model max_position_embeddings is unexpectedly low!")

    # Check if model supports higher token limits
    print("\nChecking if model parameters allow long sequences...")
    if model.config.max_position_embeddings < 131072:
        print(f"ERROR: Model max position ({model.config.max_position_embeddings}) is lower than expected (131072).")

    return model

def check_model_files():
    import os
    print("\nChecking model directory structure...")
    
    model_files = os.listdir(MODEL_PATH)
    print(f"Model files in {MODEL_PATH}: {model_files}")

    required_files = ["config.json", "generation_config.json", "tokenizer.json", "pytorch_model.bin.index.json"]
    missing_files = [f for f in required_files if f not in model_files]

    if missing_files:
        print(f"ERROR: Missing expected model files: {missing_files}")
    else:
        print("All expected model files are present.")

if __name__ == "__main__":
    print("=== MODEL DEBUGGING START ===")
    
    tokenizer = check_tokenizer()
    model = check_model()
    check_model_files()
    
    print("=== MODEL DEBUGGING COMPLETE ===")
