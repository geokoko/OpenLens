from gpt4all import GPT4All
import random

local_path = "/home/geokoko/.cache/gpt4all/mistral-7b-openorca.gguf2.Q4_0.gguf"
try:
    model = GPT4All(model_name = "mistral-7b-openorca.gguf2.Q4_0.gguf", model_path = "/home/geokoko/.cache/gpt4all/")
    print(f"Model loaded successfully from {local_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

available_emotions = ['happy', 'sad', 'angry', 'surprise', 'disgust', 'fear']
emotion = random.choice(available_emotions) # here should be the emotion of the person you are talking to, evaluated by the model
# for now, we are just randomly choosing an emotion
input_text = "I am feeling very happy today. I just got a new job!"

try:
    prompt = f"The person I am having a conversation with is feeling {emotion}. This is what he said: {input_text}. Can you help me respond?"
    prompt2 = "Hello, i am sad. What can i do to feel better?"
    output = model.generate(prompt2)
    print("Prompt:", prompt2)
    print("Output:", output)
except Exception as e:
    print(f"Error generating output: {e}")