# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


import joblib

# %%
train_df = pd.read_csv(r"C:\Users\Rohan\Documents\Research\Other\Natural Language Processing\train.csv")

# %%
train_df

# %%
train_df.info()

# %%
train_df.isnull().sum()

# %%
train_df = train_df.drop(["id"], axis=1)

# %%
train_df

# %%
temp_1 = train_df['malignant'].apply(lambda x: "Not Malignant" if x < 0.5 else "Malignant")    
temp_2 = train_df['highly_malignant'].apply(lambda x: "Not Highly Malignant" if x < 0.5 else "Highly Malignant")    
temp_3 = train_df['loathe'].apply(lambda x: "Not Loathe" if x < 0.5 else "Loathe")    
temp_4 = train_df['rude'].apply(lambda x: "Not Rude" if x < 0.5 else "Rude")    
temp_5 = train_df['abuse'].apply(lambda x: "Not Abuse" if x < 0.5 else "Abuse")    
temp_6 = train_df['threat'].apply(lambda x: "Not Threat" if x < 0.5 else "Threat")    

# %% [markdown]
# _Malignant Plot_

# %%
temp_flattened = np.ravel(temp_1)
data = pd.DataFrame({'target': temp_flattened})
total = len(temp_flattened)

fig, ax = plt.subplots(figsize=(6,5))
data['target'].value_counts().plot(kind='bar', color=['seagreen', 'brown'])
ax.set_xlabel('Category')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="right")
ax.set_title('Count of Malignant comments')

for p in ax.patches:
    # Get height.
    height = p.get_height()
    # Plot at appropriate position.
    ax.text(p.get_x() + p.get_width()/2.0, height + 3, '{:1.2f}%'.format(100*height/total), ha='center')
plt.show()

# %% [markdown]
# _Highly-Malignant Plot_

# %%
temp_flattened = np.ravel(temp_2)
data = pd.DataFrame({'target': temp_flattened})
total = len(temp_flattened)


fig, ax = plt.subplots(figsize=(6,5))
data['target'].value_counts().plot(kind='bar', color=['seagreen', 'brown'])
ax.set_xlabel('Category')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="right")
ax.set_title('Count of Highly-Malignant comments')

for p in ax.patches:
    # Get height.
    height = p.get_height()
    # Plot at appropriate position.
    ax.text(p.get_x() + p.get_width()/2.0, height + 3, '{:1.2f}%'.format(100*height/total), ha='center')
plt.show()

# %% [markdown]
# _Loathe Plot_

# %%
temp_flattened = np.ravel(temp_3)
data = pd.DataFrame({'target': temp_flattened})
total = len(temp_flattened)



fig, ax = plt.subplots(figsize=(6,5))
data['target'].value_counts().plot(kind='bar', color=['seagreen', 'brown'])
ax.set_xlabel('Category')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="right")
ax.set_title('Count of Loathe comments')

for p in ax.patches:
    # Get height.
    height = p.get_height()
    # Plot at appropriate position.
    ax.text(p.get_x() + p.get_width()/2.0, height + 3, '{:1.2f}%'.format(100*height/total), ha='center')
plt.show()

# %%
temp_flattened = np.ravel(temp_4)
data = pd.DataFrame({'target': temp_flattened})
total = len(temp_flattened)


fig, ax = plt.subplots(figsize=(6,5))
data['target'].value_counts().plot(kind='bar', color=['seagreen', 'brown'])
ax.set_xlabel('Category')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="right")
ax.set_title('Count of Rude comments')

for p in ax.patches:
    # Get height.
    height = p.get_height()
    # Plot at appropriate position.
    ax.text(p.get_x() + p.get_width()/2.0, height + 3, '{:1.2f}%'.format(100*height/total), ha='center')
plt.show()

# %% [markdown]
# _Abuse Plot_

# %%
temp_flattened = np.ravel(temp_5)
data = pd.DataFrame({'target': temp_flattened})
total = len(temp_flattened)


fig, ax = plt.subplots(figsize=(6,5))
data['target'].value_counts().plot(kind='bar', color=['seagreen','brown'])
ax.set_xlabel('Category')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="right")
ax.set_title('Count of Abuse comments')

for p in ax.patches:
    # Get height.
    height = p.get_height()
    # Plot at appropriate position.
    ax.text(p.get_x() + p.get_width()/2.0, height + 3, '{:1.2f}%'.format(100*height/total), ha='center')
plt.show()

# %% [markdown]
# _Threat Plot_

# %%
temp_flattened = np.ravel(temp_6)
data = pd.DataFrame({'target': temp_flattened})
total = len(temp_flattened)


fig, ax = plt.subplots(figsize=(6,5))
data['target'].value_counts().plot(kind='bar', color=['seagreen', 'brown'])
ax.set_xlabel('Category')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="right")
ax.set_title('Count of Threat comments')

for p in ax.patches:
    # Get height.
    height = p.get_height()
    # Plot at appropriate position.
    ax.text(p.get_x() + p.get_width()/2.0, height + 3, '{:1.2f}%'.format(100*height/total), ha='center')
plt.show()

# %% [markdown]
# _Data Cleaning_

# %%
# Replace email addresses with 'email'
train_df['comment_text'] = train_df['comment_text'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress', regex=True)

# Replace URLs with 'webaddress'
train_df['comment_text'] = train_df['comment_text'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress', regex=True)

# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
train_df['comment_text'] = train_df['comment_text'].str.replace(r'£|\$', 'dollers', regex=True)
    
# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
train_df['comment_text'] = train_df['comment_text'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumber', regex=True)
   
# Replace numbers with 'number'
train_df['comment_text'] = train_df['comment_text'].str.replace(r'\d+(\.\d+)?', 'number', regex=True)
# Remove punctuation
train_df['comment_text'] = train_df['comment_text'].str.replace(r'[^\w\d\s]', ' ', regex=True)

# Replace whitespace between terms with a single space
train_df['comment_text'] = train_df['comment_text'].str.replace(r'\s+', ' ', regex=True)

# Remove leading and trailing whitespace
train_df['comment_text'] = train_df['comment_text'].str.replace(r'^\s+|\s+?$', '', regex=True)

# %% [markdown]
# _Malignant_

# %%
hams = train_df['comment_text'][train_df['malignant']==1]
spam_cloud = WordCloud(width=600, height=400, background_color = 'black', max_words=50).generate(''. join(hams))
plt.figure(figsize=(6,6), facecolor='k')
plt.imshow(spam_cloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

# %%
cols_target = ['malignant','highly_malignant','rude','threat','abuse','loathe']
df_distribution = train_df[cols_target].sum()\
                            .to_frame()\
                            .rename(columns={0: 'count'})\
                            .sort_values('count')

df_distribution.plot.pie(y='count',
                                      title='Label distribution over comments',
                                      figsize=(5, 5))\
                            .legend(loc='center left', bbox_to_anchor=(1.3, 0.5))

# %%
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# %%
import nltk
nltk.download('wordnet')

# %%
for i in range(len(train_df['comment_text'])):
    train_df = train_df.assign(comment_text=train_df['comment_text'].str.lower())
    k = []
    for word in train_df['comment_text'][i].split():
        k.append(lemmatizer.lemmatize(word, pos='v'))
        train_df = train_df.assign(comment_text=train_df['comment_text'].str.join(''))

# %%
x = train_df['comment_text']
y = train_df[train_df.columns[2]].values

# %%
comment = train_df['comment_text']

# %%
naive = MultinomialNB()
tf_vec = TfidfVectorizer()
x=tf_vec.fit_transform(comment)

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
naive.fit(x_train, y_train)

# %%
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# %%
def calculate_metrics(y_true, y_pred):

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
    }

    return metrics

# %%
y_pred = naive.predict(x_test)

# %%
metrics = calculate_metrics(y_test, y_pred)

print("Accuracy:", metrics["accuracy"])
print("F1 score:", metrics["f1_score"])
print("Precision:", metrics["precision"])
print("Recall:", metrics["recall"])

# %%
joblib.dump(y_pred, "model")

# %%
confusion_matrix_np = confusion_matrix(y_test, y_pred)
num_classes = 2
# Create a heatmap for the confusion matrix
plt.figure(figsize=(20, 16))
sns.heatmap(confusion_matrix_np, annot=True, fmt='g', cmap='rocket_r',
            xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# %%
import pickle
with open('model1.pkl', 'wb') as f:
    pickle.dump(naive, f)


