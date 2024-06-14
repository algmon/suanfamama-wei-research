# Scalable Diffusion Models with Transformers (DiT)
## Author
* Wei Jiang, Suanfamama, wei@suanfamama.com
* Mama Xiao, Suanfamama, mama.xiao@suanfamama.com

## Key Points
* We do NOT use U-Net anymore but transformers
* USE transformers as the backbone
* latent diffusion models are NOT only for images, but for multimodal
* We should focus more on efficiently process high-dimensional data

## Contributions
1. **Transformer Backbone**: The authors propose using transformers as the backbone for diffusion models, replacing the commonly-used U-Net architecture. This approach leverages the transformer's ability to handle long-range dependencies and capture complex patterns in data.

2. **Latent Diffusion Models**: They train latent diffusion models specifically designed for images. By focusing on the latent space, the model can efficiently process high-dimensional data while maintaining scalability.

3. **Scalability**: One of the primary focuses of the paper is on the scaling properties of transformers when used in the context of diffusion models. The authors demonstrate that transformers can scale effectively, making them suitable for large-scale data generation tasks.

4. **Training and Sampling**: The paper provides detailed methodologies for training and sampling with diffusion transformers. This includes the implementation of PyTorch model definitions, pre-trained weights, and code for training and sampling.

5. **Empirical Results**: The authors present empirical results showcasing the effectiveness of their approach. The results highlight the model's ability to generate high-quality images with improved scalability compared to traditional methods.

### Summary
1. **New Architecture**
2. **Efficiency**
3. **Implementation**

### Findings
The findings in this paper suggest that transformers can significantly improve the performance and scalability of diffusion models, making them a powerful tool for various generative tasks, including image and fashion video generation. This approach opens up new possibilities for creating high-quality, realistic content with greater efficiency.