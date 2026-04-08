
import nbformat as nbf
import os
import sys

nb = nbf.v4.new_notebook()

# Cells
cells = []

# Title & Description
cells.append(nbf.v4.new_markdown_cell('# Twitter Sentiment Analysis: Exploratory Data Analysis 📊'))

# Section 1: Setup
cells.append(nbf.v4.new_markdown_cell('## 1. Data Exploration\nWe visualize the class balance and analyze text patterns in our sentiment-labeled tweets.'))
cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(".."))
from src.data_loader import download_data, load_data
from src.preprocess import clean_tweet
"""))

# Download & Load (sampled for speed)
cells.append(nbf.v4.new_code_cell("""
path = download_data()
df = load_data(path)

# For EDA, sample 20,000 tweets
df_sample = df.sample(20000, random_state=42).reset_index(drop=True)

# Sentiment Mapping: 0 -> Negative, 2 -> Neutral (if any), 4 -> Positive
df_sample['sentiment'] = df_sample['target'].map({0: 'Negative', 2: 'Neutral', 4: 'Positive'})

print(f"Sample size: {len(df_sample)}")
df_sample.head()
"""))

# Sentiment Distribution
cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df_sample, palette='viridis')
plt.title('Sentiment Distribution (Sample)')
plt.show()
"""))

# Tweet lengths
cells.append(nbf.v4.new_code_cell("""
df_sample['text_length'] = df_sample['text'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(df_sample[df_sample['sentiment']=='Positive']['text_length'], color='green', label='Positive', kde=True)
sns.histplot(df_sample[df_sample['sentiment']=='Negative']['text_length'], color='red', label='Negative', kde=True)
plt.title('Tweet Length Distribution')
plt.legend()
plt.show()
"""))

nb.cells = cells
with open('notebooks/eda.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
