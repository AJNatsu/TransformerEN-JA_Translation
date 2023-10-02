# English to Japanese Transformer Translation Model

This project contains a model capable of translating text from English to Japanese. It uses a deep learning approach to learn the translation mappings between the two languages.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Training](#training)
  - [Translation](#translation)
- [Files](#files)
- [Acknowledgements](#acknowledgements)

## Project Structure

The project is structured as follows:

- `config.py`: Contains configuration settings for the model.
- `EngJp_datasets.py`: Contains functions for loading and processing the English and Japanese datasets.
- `model.py`: Contains the definition of the translation model.
- `train.py`: Contains code for training the model.
- `translate.py`: Contains code for translating English text to Japanese using the trained model.

## Getting Started

### Requirements

- Python 3.7+
- Tensorflow 2.4+
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/user/repo.git
```
2. Navigate to the project directory:
```bash
cd repo
```
### Install the dependencies:
```bash
pip install -r requirements.txt
```

### Training
To train the model, run the following command:
```bash
python train.py
```
## Translation
To translate English text to Japanese, run the following command:
``` bash
python translate.py --input "Your text here"
```
## Files
- `config.py`: Contains configuration settings for the model.
- `EngJp_datasets.py`: Contains functions for loading and processing the English and Japanese datasets.
- `model.py`: Contains the definition of the translation model.
- `train.py`: Contains code for training the model.
- `translate.py`: Contains code for translating English text to Japanese using the trained model.
Acknowledgements
- - Dataset: [Link to the dataset]([https://dataset.link](https://huggingface.co/datasets/opus100/viewer/en-ja/train)https://huggingface.co/datasets/opus100/viewer/en-ja/train)


