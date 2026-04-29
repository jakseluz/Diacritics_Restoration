# Diacritics Restoration project

TODO

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