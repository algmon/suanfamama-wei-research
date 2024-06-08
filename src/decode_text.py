# Text Decoder with * Library
"""
This module is developed by Suanfamama.

Authors:
- Wei Jiang (wei@suanfamama.com)
- Mama Xiao (mama.xiao@suanfamama.com)
"""

'''
Explanation:

1. Import necessary libraries: Import torch for tensor operations and AutoTokenizer, AutoModelForSeq2SeqLM from the transformers library.
2. Initialize TextDecoder: Create an instance of the TextDecoder class with the desired pre-trained model name or path.
3. Prepare encoded text: Create a tensor representing the encoded text. This can be obtained from a separate TextEncoder instance or from other sources.
4. Generate text (inference): Call the TextDecoder instance with the encoded text to generate text without a target. The output will be a tensor of generated token IDs.
5. Decode generated text: Use the tokenizer to decode the generated token IDs into human-readable text.
6. Prepare target text (training): Define the target text that you want the model to learn to generate.
7. Train with target text: Call the TextDecoder instance with both the encoded text and the target text to calculate the loss. This will update the model's parameters to improve its ability to generate the desired text.

Additional Notes:

* You can adjust the model_name_or_path argument to use different pre-trained models for text generation.
* The target_text argument is optional. If not provided, the model will generate text without a target (inference).
* The loss value is a measure of how well the model is performing on the given training data. You can use this value to monitor the training progress and adjust the model's hyperparameters if needed.
* Make sure we have the correct version of the transformers library installed. The AutoModelForSeq2SeqLM class was introduced in version 4.17.0.
* If you encounter any errors, check the documentation for the transformers library and the specific models you're using.
'''

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TextDecoder(nn.Module):
    def __init__(self, model_name_or_path):
        super(TextDecoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    def forward(self, encoded_text, target_text=None):
        # Generate text from the encoded representation
        if target_text is None:
            # Generate text without target (inference)
            generated_ids = self.model.generate(encoded_text)
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        else:
            # Train the model with target text
            labels = self.tokenizer.encode(target_text, return_tensors="pt")
            loss = self.model(input_ids=encoded_text, labels=labels).loss
            return loss

# Initialize the TextDecoder with the desired model
decoder = TextDecoder("t5-small")  # Or any other compatible model

# Example encoded text
encoded_text = torch.tensor([[1, 2, 3, 4, 5]])

# Generate text without target (inference)
generated_text = decoder(encoded_text)
print("Generated text:", generated_text)

# Example target text for training
target_text = "This is an example target text."

# Train the model with target text
loss = decoder(encoded_text, target_text)
print("Loss:", loss)

'''
Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
/usr/local/Caskroom/miniforge/base/envs/algmon/lib/python3.11/site-packages/transformers/generation/utils.py:1133: UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.
  warnings.warn(
Generated text: None
Loss: tensor(5.1378, grad_fn=<NllLossBackward0>)
'''