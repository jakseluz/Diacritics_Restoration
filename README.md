# Diacritics Restoration project

## Introduction to the problem

The project focuses on diacritics restoration in Polish language words taking the context into account.

e.g. Labuz -> Łabuz


### Research
Articles which I found to be adequate for the problem:
- „Diacritics Restoration Using Neural Networks”\
(Jakub N´aplava, Milan Straka, Pavel Straˇn´ak, Jan Hajiˇc, 2018)
- [„Diacritics Restoration using BERT with Analysis on Czech language”\
(Jakub N´aplava, Milan Straka, Jana Strakov´a, 2021)](https://arxiv.org/abs/2105.11408)
- [„Correcting Diacritics and Typos with a ByT5 Transformer Model”\
(Lukas Stankeviˇcius, Mantas Lukoˇseviˇcius, Jurgita Kapoˇci¯ut˙e-Dzikien˙e,
Monika Briedien˙e, Tomas Krilaviˇcius, 2022)](https://arxiv.org/abs/2201.13242)
- [„Dilated Convolutional Neural Networks for Lightweight Diacritics
Restoration”\
(B´alint Csan´ady, Andr´as Luk´acs, 2022)](https://arxiv.org/abs/2201.06757)
- [„Romanian Diacritics Restoration Using Recurrent Neural Networks”\
(Stefan Ruseti, Teodor-Mihai Cotet, and Mihai Dascalu, 2020)](https://arxiv.org/abs/2009.02743).


### Main possible approaches
- character-level classification
- transformers connected with an external LLM
- sequence-to-sequence.


### Project assumptions
- self-supervised learning
- batch generating during the learning process - by diacritics removal.


### Dataset I used
- Polish Wikipedia, using [datasets library](https://huggingface.co/docs/datasets/index) - large and fully sufficient for learning.
Wikimedia Wikipedia (PL):
[https://huggingface.co/datasets/wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia):
    ```python
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.pl")
    ```


### Other datasets - promising but not needed here:
- CulturaX (Polish subset):
https://huggingface.co/datasets/uonlp/CulturaX
- hand-annotated million NJKP corpus:
https://nkjp.pl/index.php?page=14lang=0
- CLARIN-PL corpuses:
https://clarin-pl.eu/catalog/resources - e.g. Parliamentary sessions of Sejm & Senat RP (300 milion of
tokens)
- PolEval (NLP competitions):
http://poleval.pl/ - e.g. to compare used data with competitors solutions.


### Metrics for evaluation
- ~~accuracy~~ - not especially helpful - can be good even when a model does not work (diacritics percentages in words are quite low)
- WER (Word Error Rate) - mistaken word percentage 
- CER (Character Error Rate) - mistaken character percentage
- DER (Diacritic Error Rate) - mistaken diacritics percentage.

From the above, I have chosen CER to be the most valuable indicator.
That is beacuse models can - not only restore diacritics where they are expected to do it - but also where the letter should be untouched.
CER takes into account both situations and present the general model efficiency when considering the project topic.

## Requirements

- Python 3.12\
  [file with requirements](./requirements.txt)\
  [environment](./environment.yml)

Install dependencies with:
--

Project initialised using conda environment. In order to create the one, run:
```bash
conda create --name <env> --file requirements.txt
```
or
``` bash
conda env create -n <env> -f environment.yml
```
You can also probably work with other tools like pip:
``` bash
pip install -r requirements.txt
```
PS: In order not to have problems with your local CUDA configuration etc., please check [Pytorch prerequisites](https://pytorch.org/get-started/locally/) and copy a command to install Pytorch properly.

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/jakseluz/Diacritics_Restoration.git
   cd Diacritics_Restoration
   ```

2. Check **notebooks/[main.ipynb](./notebooks/main.ipynb) notebook file** for the tutorial.

## Author

- Jakub Łabuz ([jakseluz](https://github.com/jakseluz))

---

_For questions or contributions, please open an issue or submit a pull request!_