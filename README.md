
# iTransformer: Custom Transformer Model in PyTorch

This repository contains the implementation of a custom Transformer model named **iTransformer**, developed using PyTorch. This project is part of HW3 for the Deep Learning course instructed by Dr. Soleymani.

## Project Overview

The goal of this project is to build and train a Transformer model from scratch. The iTransformer model is designed to handle various sequential data tasks such as language modeling, sequence-to-sequence prediction, or translation tasks. Transformers are highly effective due to their attention mechanisms, allowing the model to focus on different parts of the input sequence.

### Key Features:
- **Custom Transformer Architecture**: Built from scratch using PyTorch.
- **Multi-Head Self-Attention**: Implemented to capture dependencies between different positions in the input sequence.
- **Feedforward Neural Networks**: Stacked layers used to transform and process the information from the attention mechanism.
- **Positional Encoding**: Encodes the position of input tokens as Transformers do not inherently account for token order.
- **Configurable and Extendable**: Designed to be easily customizable with adjustable layers, hidden sizes, and attention heads.

## Model Architecture

The iTransformer follows the architecture of the classic Transformer model:

1. **Embedding Layer**: Converts input tokens into dense vectors.
2. **Positional Encoding**: Adds positional information to the embeddings to help the model understand the order of tokens in a sequence.
3. **Multi-Head Self-Attention Mechanism**: This mechanism allows the model to focus on different parts of the input sequence by processing different positions simultaneously.
4. **Feedforward Networks**: Applied after the attention layers to further transform the sequence representations.
5. **Output Layer**: Produces the final predictions based on the transformed representations.

### Model Configurations

- **Number of Layers**: Configurable, default is set to 6 layers (encoder and decoder).
- **Attention Heads**: The default configuration uses 8 heads in multi-head attention layers.
- **Hidden Units**: The hidden size is set to 512 units, adjustable for experimentation.
- **Dropout**: A dropout layer is included to prevent overfitting during training.

## Dataset and Task

The iTransformer model can be applied to various datasets involving sequential data, such as text datasets for language modeling or machine translation tasks. The specific dataset used in this project can be adapted as needed. Typically, the data would be tokenized and preprocessed before being fed into the model.

### Example Dataset

For language modeling or translation tasks, you can use popular datasets like:
- **WMT English-German Dataset** for machine translation.
- **Penn Treebank Dataset** for language modeling.

The dataset should be tokenized, and padding may be applied to ensure consistent input sizes.

## Installation and Setup

To run the project:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/AqaPayam/iTransformer_PyTorch.git
    ```

2. **Install Dependencies**:
    You need to have Python and the following libraries installed:
    - PyTorch
    - Numpy
    - Pandas
    - Matplotlib

    Install the dependencies with:
    ```bash
    pip install torch numpy pandas matplotlib
    ```

3. **Run the Jupyter Notebook**:
    Open and run the iTransformer notebook:
    ```bash
    jupyter notebook iTransformer_Model.ipynb
    ```

## Training and Evaluation

### Training the Model

- **Training Process**: The iTransformer is trained using backpropagation. The model learns to minimize the loss between its predictions and the actual target sequences using a variant of the Adam optimizer.
- **Loss Function**: Cross-entropy loss is typically used for sequence-to-sequence tasks such as language modeling or translation.
- **Batch Size and Epochs**: The notebook allows configuring the batch size and number of epochs. Default values are set to 64 and 10 epochs, respectively.

### Evaluation Metrics

- **Perplexity**: Commonly used for language modeling to measure the uncertainty of the model.
- **Accuracy**: For tasks like sequence classification or translation, accuracy can be used to evaluate performance.

## Customization

The iTransformer is designed to be flexible. You can customize the model by:
- **Changing the Number of Layers**: Adjust the number of encoder and decoder layers.
- **Modifying the Number of Attention Heads**: Experiment with different numbers of attention heads for multi-head attention layers.
- **Adjusting Hidden Units**: Change the size of the hidden layers for different tasks.

### Example Use Cases

The iTransformer can be adapted for various tasks, including:
- **Language Modeling**: Predicting the next word in a sequence.
- **Machine Translation**: Translating text from one language to another.
- **Sequence Classification**: Classifying sequences based on learned representations.

## Visualization

The notebook provides tools for visualizing training progress:
- **Training and Validation Loss**: Plotted over time to monitor model performance.
- **Attention Weights**: Visualizations of the attention distributions can be included to understand which parts of the input the model focuses on during predictions.

## Conclusion

The iTransformer project demonstrates the flexibility and power of Transformer models for handling sequential tasks. By building the model from scratch in PyTorch, this project provides insights into the underlying mechanics of the Transformer architecture, making it a valuable learning resource.

## Acknowledgments

This project is part of a deep learning course by Dr. Soleymani.
