# ell-studio --storage .\logdir

import os
from groq import Groq
import ell
from PIL import Image
import base64
from ell.types.message import ImageContent

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
# Get the API keys from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")

# Set up the Ell Groq client and model
ell.init(verbose=True,store="./logdir")
ell.models.groq.register(api_key=groq_api_key)
vision_model = 'llama-3.2-90b-vision-preview'
text_model="llama-3.1-70b-versatile"

# Function to analyze the image using Llama 3.2 Vision

# @ell.simple(model=vision_model)
# def analyze_image(base64_image, user_prompt):
#     return f"Describe all elements in the image and their relative positions"

image_path = r"C:\Users\Henrique\Desktop\stuff.jpeg"

@ell.simple(model=vision_model)
#def describe_image(image_url: str):
def describe_image(image_path: str):

    image = Image.open(image_path)
    
    # Getting the base64 string
    #base64_image = encode_image(image_path)

    return [
        #ell.system("You are a helpful assistant that describes images."),
        ell.user(["Describe all items in the image, what is written on their labels and their relative positions", image])
        #ell.user(["Describe all elements in the image and their relative positions.", ImageContent(url=image_url, detail="low")])
    ]

description = describe_image(image_path)
print(description)

@ell.simple(model=text_model)
def object_finder(description : str, object : str):
    ell.system("You are a helpful assistant that helps the user find objects based on a scene description.")
    ell.user(f"Here is a description of a scene: {description}")
    return f"Explain to the user how to reach the following item: {object}."

results = object_finder(description, "Curry powder")