from gpt4all import GPT4All

model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
output = model.generate("The person I am having a conversation is feeling sad. He is saying I am really sad I lost my job. What should I answer?")

print(output)