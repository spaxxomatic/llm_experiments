# Example showcasing the impact of batched generation. Measurement device: RTX3090
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to("cuda")
inputs = tokenizer(["I want to travel to "], return_tensors="pt").to("cuda")

def print_tokens_per_second(batch_size):
    new_tokens = 100
    cumulative_time = 0

    # warmup
    model.generate(
        **inputs, do_sample=True, max_new_tokens=new_tokens, num_return_sequences=batch_size
    )

    for _ in range(10):
        start = time.time()
        output = model.generate(
            **inputs, do_sample=True, max_new_tokens=new_tokens, num_return_sequences=batch_size
        )
        print(output)
        cumulative_time += time.time() - start
    print(f"Tokens per second: {new_tokens * batch_size * 10 / cumulative_time:.1f}")

print_tokens_per_second(1)   # Tokens per second: 418.3
print_tokens_per_second(64)  # Tokens per second: 16266.2 (~39x more tokens per second)
