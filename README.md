# Repo of master thesis




## Workflow

Intermediary data is not stored on this repo and has to be generated. Run the following files if this is your first time:

1. `checks.ipynb` to create folder structure (where the following scripts will save their data)
2. `Eligibility Criteria.ipynb` to generate list of eligible patient ids
3. `Generating Dataset Pipeline.ipynb` to generate a json dataset of the admissions data
4. `Split dataset.ipynb` to generate the train-validation-test splits.
5. `Register best model.ipynb` to create, train and save a model's weights and hyperparameters
6. `Create Variational Outputs.ipynb` to use the previous model (or any other really) to generate several forward passes on the validation set

Having done this the analysis notebooks can work:
1. `Accuracy as a function of confidence.ipynb` for the general uncertainty analysis
2. TODO

### Setup
To be able to run and reproduce this repo, create a conda environment:
```bash
conda env create -n <env_name> -f environment.yml
```
