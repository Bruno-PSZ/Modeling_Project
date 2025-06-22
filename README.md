# Evaluating the importance of input data representation for

This repository contains the code and materials used in my project for classifying 28 different cell types from single-cell RNA sequencing (scRNA-seq) data using deep learning techniques. The project compares models trained on raw gene expression data versus embeddings generated from scVI and scANVI models.

##  Setup
1. Install dependencies with:

```bash
pip install -r requirements.txt
```
2. Download raw and preprocessed data from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE155249 and put it into /DATA folder. Preprocessed data should be saved as DATA\GSE155249_main and raw data as Data/GSE155249_RAW.h5ad.

3. Crate secrets.env file in the root of the project. This file is used for storing the API keys for ClearML. Your file should look like this:

```bash
CLEARML_API_ACCESS_KEY=ACCESS_KEY(put yours)
CLEARML_API_SECRET_KEY=SECRET_KEY(put yours)
```

4. This project uses https://clear.ml/ for logging the results of all the experiments. I recommend setting up the free account there to make it easier to reproduce the code. It is possible to run code without the ClearML account but it requires some manula changes

5. The folder Results_JSON contains results from the experiments downloaded from ClearML
