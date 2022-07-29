# %% [markdown]
# Import all necessary libraries and install everything you need for training:

# %%
# install the libraries necessary for data wrangling, prediction and result analysis
import json
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score,precision_score, recall_score
import torch
from numba import cuda
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

# %%
# Install transformers
# (this needs to be done on Kaggle each time you start the session)
#!pip install -q transformers

# %%
# Install the simpletransformers
#!pip install -q simpletransformers
from simpletransformers.classification import ClassificationModel

# %%
# Install wandb
#!pip install -q wandb

# %%
import wandb

# %%
# Login to wandb
wandb.login()

# %%
# Clean the GPU cache


# %% [markdown]
# ### Import the data

# %%
# FTD
train_df = pd.read_csv("data/FTD-train.txt", sep="\t", index_col=0)
dev_df = pd.read_csv("data/FTD-dev.txt", sep = "\t", index_col = 0)
test_df = pd.read_csv("data/FTD-test.txt", sep = "\t", index_col = 0)

print("FTD train shape: {}, Dev shape: {}, Test shape: {}.".format(train_df.shape, dev_df.shape, test_df.shape))

# %%
train_df.head()

# %% [markdown]
# ## Testing

# %% [markdown]
# We will use the multilingual XLM-RoBERTa model
# https://huggingface.co/xlm-roberta-base

# %%
# Create a file to save results into (you can find it under Data: Output). Be careful, run this step only once to not overwrite the results file.
#results = []

#with open("results/FTD-Experiments-Results.json", "w") as results_file:
#    json.dump(results,results_file, indent= "")

# %%
# Open the main results file:


# %%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
# Create a list of labels
LABELS = train_df.labels.unique().tolist()
LABELS

# %%
# Initialize Wandb
run = wandb.init(project="FTD-learning-manual-hyperparameter-search", entity="tajak", name="testing-trained-model")

# %%
# Load the trained model from Wandb
model_name = "FTD-classifier"
# Use the latest version of the model
model_at = run.use_artifact(model_name + ":latest")
# Download the directory
model_dir = model_at.download()

# Loading a local save
model = ClassificationModel(
    "xlmroberta", model_dir)


# %%
# Update the args
model = ClassificationModel(
    "xlmroberta", model_dir, args= {"silent":True})

# %%
# Let's change the numeric values into nominal so that the confusion matrix is more informative
# here's the dictionary:
FTD_mapping = {0: 'A1 (argumentative)', 1: 'A11 (personal)', 2: 'A12 (promotion)', 3: 'A14 (academic)', 4: 'A16 (information)', 5: 'A17 (review)', 6: 'A4 (fiction)', 7: 'A7 (instruction)', 8: 'A8 (news)', 9: 'A9 (legal)'}

# %%
# Map the labels
LABELS_mapped = [FTD_mapping[x] for x in LABELS]
LABELS_mapped

# %%
# Create a function to predict
def predict_FTD(df, new_file_name):
    """
    This function takes a dataset and applies the trained model on it to infer predictions.
	It returns and saves the resulting df with added columns with FTD predictions.

    Args:
    - df: dataframe on which we want to apply prediction. The text should be in a column "text".
    - new_file_name: define the name of the new file
    """
    # Predict on the df
    def make_prediction(input_string):
        return model.predict([input_string])[0][0]

    y_pred = df.text.apply(make_prediction)

    # Map the numeric categories, used for prediction, to nominal
    y_pred_mapped = y_pred.map(FTD_mapping)
    df["FTD_pred"] = y_pred_mapped

    # Save the new dataframe which contains the y_pred values as well
    df.to_csv(f"{new_file_name}", sep="\t")

    return df
# %%
# Import data about CORE
core_df = pd.read_csv("Data-Preparation/data/CORE-all-information.csv", sep="\t", index_col=0)

core_df

# %%
core_df = predict_FTD(core_df, "final_data/CORE-all-information-FTD-predicted.csv")


