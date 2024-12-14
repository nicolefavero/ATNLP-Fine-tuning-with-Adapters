# Group Reimplementation Project: Transformer-Based SCAN Task

This repository is a reimplementation of experiments from the paper [*Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks*](https://arxiv.org/abs/1711.00350) by Brendan Lake and Marco Baroni. Instead of using RNNs, GRUs, or LSTMs as in the original paper, we implement a Transformer-based model to replicate and analyze the results of Experiments 1, 2, and 3.

## Introduction
The goal of this project is to evaluate the compositional generalization capabilities of Transformer models on the SCAN dataset. SCAN is a synthetic dataset that pairs commands (e.g., "walk twice and jump") with corresponding actions (e.g., "WALK WALK JUMP"). We test the ability of our model to generalize across three key splits:
- **Experiment 1**: Simple split with varying training data sizes.
- **Experiment 2**: Length-based split.
- **Experiment 3**: Compositional split.

This project is designed for educational purposes.


## Dependencies
The project is implemented in Python using PyTorch. Ensure you have the following dependencies installed:
- Python 3.8+
- PyTorch 1.13+
- TorchMetrics
- scikit-learn
- tqdm
- numpy


## Code Structure
This repository contains the following components:
- **`dataset.py`**: A custom data loader for the SCAN dataset, including vocabulary creation and text-token transformations.
- **`model/transformer.py`**: An implementation of a Transformer-based sequence-to-sequence model, including multi-head attention, positional encodings, encoder-decoder layers, and mask generation.
- **`train.py`**: The training and evaluation script for the model, including teacher forcing, greedy search, and oracle decoding.
- **`data/`**: Directory for SCAN dataset files. Use the preprocessed dataset from [Transformer-SCAN](https://github.com/jlrussin/transformer_scan).
- **`model/`**: Directory for saving trained model checkpoints.


## Data Input
The SCAN dataset is sourced from the [Transformer-SCAN repository](https://github.com/jlrussin/transformer_scan). Each dataset split consists of text files with lines in the format:
```
IN: <COMMAND> OUT: <ACTION>
```
Example:
```
IN: jump thrice OUT: JUMP JUMP JUMP
IN: turn left twice OUT: LTURN LTURN
```

The dataset is tokenized, and both source (commands) and target (actions) vocabularies are built dynamically. Special tokens such as `<PAD>`, `<UNK>`, `<BOS>`, and `<EOS>` are used for training and evaluation.


## Usage
1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo-name
    cd your-repo-name
    ```

2. **Download the SCAN dataset**:
    Place the dataset files in the `data/` directory. Example structure:
    ```
    data/
      length_split/
        tasks_train_length.txt
        tasks_test_length.txt
      simple_split/
        size_variations/
          tasks_train_simple_p1.txt
          tasks_test_simple_p1.txt
          ...
    ```

3. **Train the model**:
    Use the `main` function in `train.py` to train on a specific dataset split. For example:
    ```bash
    python train.py
    ```

    Key parameters for training include:
    - `train_path`: Path to training dataset.
    - `test_path`: Path to testing dataset.
    - `random_seed`: Random seed for reproducibility.
    - `oracle`: Enable oracle decoding (default: False).

4. **Evaluate the model**:
    After training, the model evaluates performance using:
    - Teacher forcing.
    - Greedy search decoding.
    - Oracle decoding (if enabled).


## Evaluation
### Metrics
The following metrics are used to evaluate model performance:
- **Token-level accuracy**: Measures the percentage of correct token predictions.
- **Sequence-level accuracy**: Measures whether the entire output sequence matches the target sequence.

### Example Output
```plaintext
Dataset p16 - Epoch: 10
Train Loss: 0.1234
Test Loss: 0.4567
Greedy Search Loss: 0.3456
Accuracy: 92.34%, Sequence Accuracy: 85.67%
```


## Results
We aim to reproduce and compare the following key findings:
- Performance of Transformers across dataset splits and sizes.
- Ability of Transformers to generalize to longer or unseen command sequences.

You can use the `run_all_variations()` function in `train.py` to evaluate the model on all dataset splits and sizes.


## Acknowledgments
This project is inspired by the experiments conducted in [Lake & Baroni, 2017](https://arxiv.org/abs/1711.00350). We thank [Transformer-SCAN](https://github.com/jlrussin/transformer_scan) for providing preprocessed SCAN datasets and code references.
