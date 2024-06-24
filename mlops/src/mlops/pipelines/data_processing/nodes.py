from typing import Dict, Tuple

import pandas as pd
from typing import Dict, Tuple
from kedro.io import MemoryDataset
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  # we only need pyplot
sns.set()  # set the default Seaborn style for graphics


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

# function to remove stopwords


def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

# Clean Text


def clean_text(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = ' '.join(text.split())
    return text

# Stemming


def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

# preprocess description column on cleaning and stemming


def preprocess_descriptions(train_data: pd.DataFrame):
    # Ensure all values in 'Description' are strings and handle NaN values
    train_data['Description'] = train_data['Description'].astype(
        str).fillna('')

    # Apply your preprocessing functions
    train_data['Description'] = train_data['Description'].apply(
        lambda x: remove_stopwords(x))
    train_data['Description'] = train_data['Description'].apply(
        lambda x: clean_text(x))
    train_data['Description'] = train_data['Description'].apply(stemming)

    return train_data


def one_hot_encode_column(df: pd.DataFrame, description_col: str, target_col: str) -> pd.DataFrame:
    rows = []
    unique_values = df[target_col].unique()
    descriptions = df[description_col].unique()
    for description in descriptions:
        row = {'Description': description}
        for unique_value in unique_values:
            row[unique_value] = 1 if unique_value in df[df['Description']
                                                        == description][target_col].values else 0
        rows.append(row)
    new_df = pd.DataFrame(rows, columns=['Description']+list(unique_values))
    return new_df


def create_encoded_df(df: pd.DataFrame, description_col: str, target_col: str):
    unique_departments = df['Department'].unique()
    partitioned_data = {}
    for department in unique_departments:
        df_department = df[df['Department'] == department]
        encoded_df = one_hot_encode_column(
            df_department, description_col, target_col)
        partitioned_data[department] = encoded_df
    return partitioned_data


def create_department_encoded_df(df: pd.DataFrame) -> dict:
    return one_hot_encode_column(df[['Description', 'Department']], 'Description', 'Department')


def create_techgroup_encoded_df(df: pd.DataFrame) -> dict:
    return create_encoded_df(df, description_col="Description", target_col="Tech Group")


def create_subcategory_encoded_df(df: pd.DataFrame) -> dict:
    return create_encoded_df(df, description_col="Description", target_col="Sub-Category")


def create_category_encoded_df(df: pd.DataFrame) -> dict:
    return create_encoded_df(df, description_col="Description", target_col="Category")


def merge_datasets(department_df: pd.DataFrame, category_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(department_df, category_df,
                         on='Description', how='inner')
    return merged_df
