# Image Captioning using CNN and RNN

This project implements an image captioning model that integrates Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to generate natural language descriptions for images. The model uses a pre-trained VGG16 CNN to extract image features and an LSTM-based RNN decoder with attention mechanisms (soft and hard) to generate the corresponding captions.

## Introduction

Image captioning is a task at the intersection of Computer Vision and Natural Language Processing. It involves analyzing an image to understand its content and generating a relevant textual description. This project aims to develop a model capable of this task by leveraging CNNs for feature extraction and RNNs for sequential caption generation.

**Applications**:
- Enhancing accessibility for visually impaired individuals.
- Automated content management and image indexing.
- Use in dynamic systems like real-time news captioning and surveillance.

## Project Overview

- **Dataset**: Flickr8k
- **Split**: 80% Training / 20% Testing
- **CNN Encoder**: VGG16 (pre-trained)
- **RNN Decoder**: LSTM
- **Attention Mechanisms**: Both Soft and Hard Attention
- **Evaluation Metric**: BLEU Score
- **Loss Function**: Cross-Entropy
- **Optimizer**: Adam
- **Epochs**: 10

## Model Architecture

### CNN Encoder (VGG16)
- Extracts detailed visual features from the input image.
- Removes the final classification layer.
- Outputs feature vectors for the image.

### RNN Decoder (LSTM)
- Accepts encoded image features as input.
- Generates sequential words to form a complete caption.

### Attention Mechanisms
- **Soft Attention**: Assigns weights to all parts of the image simultaneously.
- **Hard Attention**: Selects a specific region of the image to focus on at each step (stochastic approach).

## Experiments and Results

Three experiments were conducted:
1. **Base Model (No Attention)**  
2. **Model with Soft Attention**  
3. **Model with Hard Attention**

### Loss Comparison
- Training loss decreased consistently across epochs for all models.
- Soft and Hard attention models showed improved convergence over the base model.

### BLEU Scores

| Model                 | BLEU Score |
|----------------------|------------|
| Base                 | ~0.45      |
| Soft Attention       | ~0.50      |
| Hard Attention       | ~0.48      |

*Reference for comparison*:  
Anderson et al., *"Where to put the Image in an Image Caption Generator"* (2017)

### Qualitative Results


To illustrate the performance of our models, we present sample outputs from each experiment:

### Base Model (No Attention)

![Base Model Captions](Figures/base_captions.png)

The base model generates captions by relying solely on the encoded image features without any attention mechanism. While it captures the general context, it may miss finer details due to the lack of focused attention.

### Soft Attention Model

![Soft Attention Captions](Figures/soft_attention_captions.png)

Incorporating soft attention allows the model to weigh different parts of the image when generating each word in the caption. This results in more contextually relevant and detailed descriptions, as the model can focus on salient regions of the image.

### Hard Attention Model

![Hard Attention Captions](Figures/hard_attention_captions.png)

The hard attention mechanism enables the model to select specific regions of the image to focus on at each time step. This stochastic approach can lead to more precise captions but may introduce variability due to its sampling nature.

These qualitative examples demonstrate the improvements in caption generation when employing attention mechanisms, with soft attention providing smoother and more consistent enhancements over the base model.


## Required Libraries

Make sure to install the following Python packages:

```bash
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
nltk
Pillow
tqdm
```

---
## References and Acknowledgments

- Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). *Show and Tell: A Neural Image Caption Generator*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). [arXiv:1411.4555](https://arxiv.org/abs/1411.4555)
  
- Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhutdinov, R., Zemel, R., & Bengio, Y. (2015). *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*. Proceedings of the 32nd International Conference on Machine Learning (ICML). [arXiv:1502.03044](https://arxiv.org/abs/1502.03044)

- Flickr8k Dataset: [https://www.kaggle.com/datasets/adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)

- VGG16 Model: Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)

- We acknowledge the contributions of the PyTorch and NLTK communities for providing robust libraries.



