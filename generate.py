from transformers import pipeline
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
iret = generator("EleutherAI has", do_sample=True, min_length=50)

print (iret)
