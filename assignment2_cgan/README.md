# Assignment 2: Conditional GAN (CGAN)

## Objective
To develop a Conditional Generative Adversarial Network (CGAN) capable of generating images based on user-provided labels or categories.

## Extended Description
A Conditional Generative Adversarial Network (CGAN) is an extension of the traditional GAN architecture that incorporates additional information, such as class labels, into both the generator and discriminator. In this assignment, labels representing simple geometric shapes such as circles, squares, and triangles are provided as conditional inputs.

The generator receives a combination of random noise and class labels and produces images corresponding to the specified category. The discriminator evaluates whether the generated image is realistic and whether it matches the given label. Through adversarial training, the generator learns to create increasingly accurate images while the discriminator improves its classification ability.

This project introduces the concept of controlled image generation, where users can influence the output by specifying a desired category. The assignment demonstrates the importance of conditional inputs in generative models and provides a foundation for more advanced text-to-image generation systems.

## Tools and Technologies
- Python
- PyTorch
- NumPy
- Matplotlib

## Methodology
1. Prepare a dataset containing labeled geometric shapes.
2. Design the CGAN architecture with generator and discriminator networks.
3. Incorporate class labels as conditional inputs.
4. Train the model using adversarial learning.
5. Generate images based on selected labels.
6. Evaluate image quality and label consistency.

## Outcome
The CGAN successfully generates shape-specific images according to the provided labels, demonstrating the effectiveness of conditional image generation techniques.
