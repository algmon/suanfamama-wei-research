# Suanfamama AI Code Generator
"""
This module is developed by Suanfamama.

Authors:
- Wei Jiang (wei@suanfamama.com)
- Mama Xiao (mama.xiao@suanfamama.com)
"""

'''
Let's break down the Python code you provided, which is designed to generate code using the Gemini model from Google Vertex AI.

1. Imports and Initialization

* base64: This library is used for encoding and decoding data in base64 format, which is often used for transmitting data over the internet.
* vertexai: This is the core library for interacting with Google Vertex AI, a platform for building and deploying machine learning models.
* vertexai.generative_models: This submodule provides tools for working with generative models, like Gemini.
* vertexai.preview.generative_models: This submodule contains preview features for generative models.
* vertexai.init(project="suanfamama-fashion-gen-model", location="us-central1"): This line initializes the Vertex AI client, specifying the project ID ("suanfamama-fashion-gen-model") and the region ("us-central1") where your project is located.

2. generate Function

* model = GenerativeModel("gemini-1.5-pro-001"): This line creates an instance of the GenerativeModel class, specifically using the "gemini-1.5-pro-001" model. This model is likely a powerful language model from Google.
* responses = model.generate_content(...): This is the core function call. It uses the generate_content method of the GenerativeModel to generate text based on the provided input.

Let's break down the arguments:
* [text1]: This is the input text that you want the model to use as a prompt. In this case, it's a request for a Python program that uses Gemini to evaluate and augment fashion design requests.
* generation_config: This dictionary contains parameters that control how the model generates text.
* max_output_tokens: Limits the maximum number of tokens (words or sub-words) in the generated output.
* temperature: Controls the randomness of the generated text. Higher values lead to more creative and unpredictable outputs.
* top_p: This parameter is related to nucleus sampling, a technique for controlling the diversity of the generated text.
* safety_settings: This dictionary specifies safety settings to prevent the model from generating harmful or inappropriate content. It uses the HarmCategory and HarmBlockThreshold enums from vertexai.preview.generative_models to define the types of content to block and the severity threshold.
* stream=True: This argument enables streaming, allowing you to receive the generated text in chunks as it's being produced.
* for response in responses: print(response.text, end=""): This loop iterates through the generated responses (chunks of text) and prints them to the console. The end="" argument prevents a newline from being printed after each chunk, so the output appears as a continuous stream.

3. Input Text (text1)

text1 = """...""": This string contains the prompt that you want to use to generate the Python code. It's a request for a program that can evaluate and augment fashion design requests using Gemini.

4. Generation Configuration (generation_config)

This dictionary sets parameters for the text generation process. The values are chosen to balance creativity and control over the output.

5. Safety Settings (safety_settings)

This dictionary defines safety settings to prevent the model from generating harmful or inappropriate content. It blocks content related to hate speech, dangerous content, sexually explicit content, and harassment.

6. generate() Function Call

This line calls the generate function, which executes the code to generate the Python program based on the provided prompt and settings.

In Summary

This code demonstrates how to use the Gemini model from Google Vertex AI to generate Python code. It takes a prompt describing a desired program, sets parameters for the generation process, and includes safety settings to ensure responsible output. The code then prints the generated code to the console.

Important Notes:

* Authentication: You'll need to set up authentication for your Google Cloud project to use Vertex AI. This typically involves creating a service account and obtaining the necessary credentials.
* Model Availability: The "gemini-1.5-pro-001" model might be a preview or experimental model. Its availability and features could change over time.
* Safety Settings: The safety settings are important for ensuring responsible use of the model. You may need to adjust them based on your specific requirements and the type of content you want to generate.
'''

import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

def generate():
  vertexai.init(project="suanfamama-fashion-gen-model", location="us-central1")
  model = GenerativeModel(
    "gemini-1.5-pro-001",
  )
  responses = model.generate_content(
      [text1],
      generation_config=generation_config,
      safety_settings=safety_settings,
      stream=True,
  )

  for response in responses:
    print(response.text, end="")

text1 = """write me a python program using the latest gemini model, the input is the designer\'s request for designing clothes and decorations, the program should evaluate whether the raw text is ok to continue to the downstream or not. If not, use the llm tech to perform text argumentation in several fashion ways like style, color, material like a pro, and then output the modified text"""

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

generate()