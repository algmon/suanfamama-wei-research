# Scalable Diffusion Models with Transformers (DiT) Paper Review
## Author
* Wei Jiang, Suanfamama, wei@suanfamama.com
* Mama Xiao, Suanfamama, mama.xiao@suanfamama.com

## Key Points
1. We do NOT use U-Net as the backbone but transformers
2. latent space is NOT only for images, but for text, speech and videos as well
3. Scaling Up: Increasing model size and decreasing patch size are both effective strategies for improving the quality of generated images.
4. Optimal Configurations: Larger models (L and XL) with smaller patch sizes (2 and 4) yield the best performance in terms of FID.
5. Practical Balance: While these strategies significantly improve generative performance, they also require more computational resources. The choice of model size and patch size should balance the available computational power with the desired quality of the generated images.

## the DiT design space
The Diffusion Transformer (DiT) design space encompasses a range of architectural choices and configurations for building and training diffusion models, particularly for image data. The design space includes the following key components and variations:

* Patchify: This is the initial layer of DiT that converts a spatial input representation into a sequence of tokens. The number of tokens (T) is determined by the patch size hyperparameter (p), which can be 2, 4, or 8. A smaller patch size (p) results in more tokens and consequently higher computational requirements (Gflops), but does not significantly affect the downstream parameter count.

* DiT Block Design: After the patchification process, the input tokens are processed by a sequence of transformer blocks. These blocks are designed to handle additional conditional information such as noise timesteps, class labels, or natural language. There are four variants of transformer blocks explored in the DiT design space:

1. In-context conditioning: This approach appends vector embeddings of noise timesteps and class labels as additional tokens in the input sequence, similar to the cls tokens in Vision Transformers (ViTs). It uses standard ViT blocks without modification and introduces negligible additional computational cost.

2. Cross-attention block: This design involves concatenating the embeddings of noise timesteps and class labels into a separate sequence and modifying the transformer block to include an additional multi-head cross-attention layer. This approach adds the most computational cost to the model, with roughly a 15% overhead.

3. Adaptive Layer Norm (adaLN) block: This design replaces standard layer norm layers in transformer blocks with adaptive layer norm (adaLN). Instead of learning scale and shift parameters directly, they are regressed from the sum of the embedding vectors of noise timesteps and class labels. AdaLN is the most compute-efficient among the explored block designs and applies the same function to all tokens.

4. adaLN-Zero block: Inspired by the initialization strategy in ResNets, this design modifies the adaLN DiT block to zero-initialize certain parameters, similar to the approach in Diffusion U-Net models. It also includes regressing dimension-wise scaling parameters that are applied before any residual connections within the DiT block.

The DiT design space thus provides a variety of options to tailor the diffusion model architecture to specific requirements and computational constraints. The goal is to retain the scaling properties of the standard transformer architecture while adapting it for effective diffusion model training on image data.

## Contributions
1. **Transformer Backbone**: The authors propose using transformers as the backbone for diffusion models, replacing the commonly-used U-Net architecture. This approach leverages the transformer's ability to handle long-range dependencies and capture complex patterns in data.

2. **Latent Diffusion Models**: They train latent diffusion models specifically designed for images. By focusing on the latent space, the model can efficiently process high-dimensional data while maintaining scalability.

3. **Scalability**: One of the primary focuses of the paper is on the scaling properties of transformers when used in the context of diffusion models. The authors demonstrate that transformers can scale effectively, making them suitable for large-scale data generation tasks.

4. **Training and Sampling**: The paper provides detailed methodologies for training and sampling with diffusion transformers. This includes the implementation of PyTorch model definitions, pre-trained weights, and code for training and sampling.

5. **Empirical Results**: The authors present empirical results showcasing the effectiveness of their approach. The results highlight the model's ability to generate high-quality images with improved scalability compared to traditional methods.

### Key Findings
The findings in this paper suggest that transformers can significantly improve the performance and scalability of diffusion models, making them a powerful tool for various generative tasks, including image and fashion video generation. This approach opens up new possibilities for creating high-quality, realistic content with greater efficiency.