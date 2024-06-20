
# Generating Images from Text using GANs

This project explores the use of Generative Adversarial Networks (GANs) for generating images from text descriptions. Using TensorFlow 2 and Keras, we train a generator to create images based on the input text, which includes character content and formatting specifications such as color, font, and style.

## Project Structure
The project is divided into two main parts:

- Training a Generator Directly
- Implementing a Conditional GAN

### Part 1: Training a Generator Directly
In this phase, we use a deterministic mapping from text to images. The generator consists of two RNNs to handle variable-length text inputs. These inputs are transformed into a fixed-length encoding and then passed through a fully connected layer before being upsampled through fractional-strided convolutions to generate a 128x128x3 image. We use a tanh activation function in the final layer to ensure output values are centered around 0.

#### Loss Function
We designed a loss function tailored for the tanh activation, leveraging its gradient properties to avoid issues like vanishing or exploding gradients.

### Part 2: A Generative Adversarial Network
This section describes the transition to a GAN architecture, incorporating insights from DCGANs and conditional GANs (cGANs). We addressed challenges like mode collapse and instability, introducing techniques such as text-shuffling, minibatch discrimination, and noise addition to stabilize training.

#### Key Enhancements
-  Text Shuffling: Randomly shuffling characters and specifications to force the discriminator to learn meaningful patterns.
- Minibatch Discrimination: Enhancing the discriminator's ability to identify features across batches.
- Learning Rate Adjustments: Using a lower learning rate for the generator and a higher rate for the discriminator.
- Pretrained Text-RNN: Speeding up convergence by initializing the RNN with pretrained weights.
- Noise Addition: Adding noise to images to help the discriminator distinguish between real and fake data.
- Wasserstein GAN with Gradient Penalty: Implementing a WGAN-GP with adaptive gradients and no momentum to enhance convergence.
## Training Configuration
- Epochs: 500
- Batch Size: 16
- Optimizer: Adam with modified parameters
- β1 = 0.5, β2 = 0.9
- Learning Rates:
- Generator: 5e-5
- Discriminator: 5e-4

## Ablation Study
We performed extensive ablation studies to fine-tune the model. Key experiments included:

### Default Parameters
- Text Shuffling: Crucial for learning the vocabulary effectively.
- Minibatch Discrimination: Found to be counterproductive with text-shuffling.
- Varying Vocabulary Size: Demonstrated that smaller vocabularies accelerate training.
- Incremental Vocabulary Expansion: Improved model generalization by gradually increasing the vocabulary size.

## Results
The generated image is present in the documentation.

We observed specific challenges, such as difficulties in generating the color matplotlib-green due to the tanh's sensitivity at values around 0.
