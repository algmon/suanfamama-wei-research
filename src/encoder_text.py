# Text Encoder with Transformers Library
"""
This module is developed by Suanfamama.

Authors:
- Wei Jiang (wei@suanfamama.com)
- Mama Xiao (mama.xiao@suanfamama.com)
"""

'''
Explanation:

Import Libraries: Import torch for tensor operations and AutoTokenizer, AutoModel from the transformers library.
TextEncoder Class:
__init__: Initializes the tokenizer and model using the specified model_name_or_path.
forward:
Tokenizes the input text using the tokenizer.
Passes the tokenized input through the pre-trained model.
Extracts the last hidden state as the encoded representation.

Example Usage:
Creates an instance of TextEncoder with the desired pre-trained model.
Encodes the example text using the forward method.
Prints the encoded text representation.

Note:
This code uses a pre-trained model from the Hugging Face Transformers library. You can choose different models based on your specific needs and computational resources.

The last_hidden_state is just one possible way to extract the encoded representation. Other options might be suitable depending on the model architecture and task.

Consider exploring the documentation and examples provided by the Transformers library for more advanced usage and customization.
'''

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name_or_path):
        super(TextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, text):
        # Tokenize the text
        input_ids = self.tokenizer.encode(text, return_tensors="pt")

        # Pass through the pre-trained model
        outputs = self.model(input_ids)

        # Extract the last hidden state as the encoded representation
        encoded_text = outputs.last_hidden_state

        return encoded_text

# Example usage
input_text = "Help me to design a dress for the night party."
encoder = TextEncoder("bert-base-uncased")
encoded_text = encoder(input_text)

print("Summary")
print("Input:", input_text)
print("Output:", encoded_text)

'''
Summary
Input: Help me to design a dress for the night party.
Output: tensor([[[ 0.0445, -0.0713, -0.0701,  ..., -0.2417, -0.1862,  0.3757],
         [ 0.3245,  0.2456,  0.1511,  ...,  0.0749,  0.4048, -0.1603],
         [ 0.6843, -0.4391,  0.0850,  ..., -0.3182,  0.0185,  0.3080],
         ...,
         [ 0.3115, -0.6385,  0.6186,  ...,  0.3251, -0.3713,  0.1373],
         [-0.1733, -0.5471, -0.2973,  ...,  0.4325,  0.2267, -0.6980],
         [ 0.6650,  0.0374, -0.1604,  ...,  0.1350, -0.6131, -0.3570]]],
       grad_fn=<NativeLayerNormBackward0>)
'''