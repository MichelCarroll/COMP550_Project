import sys
from os.path import abspath, dirname, join
sys.path.append(abspath(join(dirname(__file__), '..')))

# %%
import nltk
import numpy as np
from joblib import dump
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.dataset_utils import MotleyFoolDataset

# %%
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
# %% md
# Load and prepare dataset
# %% md
# Should be `(1141, 3821, 33174)`
# %%
dataset = MotleyFoolDataset('TMF_dataset_annotated.zip')
test_set = dataset['Q1 2023':'Q4 2023']
val_set = dataset['Q1 2022':'Q4 2022']
train_set = dataset[:'Q4 2021']

print(f'''
Lengths:
========
test:       {len(test_set)}
validation: {len(val_set)}
training:   {len(train_set)}
''')

# %%
X_train_val = [instance['content'] for instance in dataset[:'Q4 2022']]
y_train_val = [instance['direction'] == 'UP' for instance in dataset[:'Q4 2022']]

X_test = [instance['content'] for instance in test_set]
y_test = [instance['direction'] == 'UP' for instance in test_set]
# %% md
# Utilities for the grid search
# %%
stopwords_set = set(stopwords.words('english'))  # set for membership check optim
lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
snowball = SnowballStemmer("english")

### Adapted from: https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
treebank_mapping = {
    'J': wordnet.ADJ,
    'V': wordnet.VERB,
    'N': wordnet.NOUN,
    'R': wordnet.ADV
}
treebank_default = wordnet.NOUN


def get_wordnet_pos(treebank_tag):
    return treebank_mapping.get(treebank_tag[0], treebank_default)


# Define custom NLTK tokenizers
def porter_tokenizer(text):
    return [porter.stem(word) for word in word_tokenize(text) if word not in stopwords_set]


def snowball_tokenizer(text):
    return [snowball.stem(word) for word in word_tokenize(text) if word not in stopwords_set]


def wordnet_lemmatizer_tokenizer(text):
    text = word_tokenize(text)
    tags = nltk.pos_tag(text)
    return [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tags
        if word not in stopwords_set
    ]


# %%
vect_parameter_grid = {
    # "vect__tokenizer": (
    #     wordnet_lemmatizer_tokenizer,
    #     porter_tokenizer,
    #     snowball_tokenizer,
    # ),
    # "vect__max_df": (0.2, 0.8),
    # "vect__ngram_range": ((1, 1)),  # unigrams or bigrams
}

vectorizer_factory = lambda: ("vect", TfidfVectorizer(
    stop_words=None,
    token_pattern=None,  # will not be used since 'tokenizer' is not None,
    tokenizer=snowball_tokenizer,
    ngram_range=(1, 1),
    max_df=0.8
))


def do_grid_search(parameter_grid, classifier=None, pipeline=None):
    print(f"Training {classifier}")
    parameter_grid |= vect_parameter_grid

    if pipeline is None:
        pipeline = Pipeline([
            vectorizer_factory(),
            ("clf", classifier),
        ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        scoring='accuracy',
        param_grid=parameter_grid,
        n_jobs=-1,
        cv=[  # Remove cross validation and use our pre-made split
            # An iterable yielding (train, test) splits as arrays of indices.
            (np.arange(*dataset.range(q_stop='Q4 2021')),
             np.arange(*dataset.range('Q1 2022', 'Q4 2022')))
        ],
        verbose=1,
    )

    grid_search.fit(X_train_val, y_train_val)

    print('Best parameters:')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameter_grid.keys()):
        print(f"{param_name}: {best_parameters[param_name]}")

    y_pred = grid_search.predict(X_test)
    print(confusion_matrix(y_test, y_pred))

    print('Accuracy score:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return grid_search


if __name__ == '__main__':
    # %% md
    # Models
    # %% md
    ## Naive Bayes
    # %%
    nb_gs = do_grid_search(
        {"clf__alpha": (0.1, 1)},
        MultinomialNB()
    )

    dump(nb_gs.best_estimator_, 'nb_gs.joblib')
    # %% md
    ## Logistic Regression
    # %%
    lf_gs = do_grid_search(
        {},
        LogisticRegression()
    )

    dump(nb_gs.best_estimator_, 'nb_gs.joblib')
    # %% md
    ## Support Vector Machine
    # %%
    svc_gs = do_grid_search(
        {
            # "clf__C": np.logspace(-3, 3, 3),
            "clf__kernel": ("linear", "poly", "rbf")
        },
        pipeline=Pipeline([
            vectorizer_factory(),
            ("lsa", TruncatedSVD(n_components=100)),
            ("norm", StandardScaler()),
            ("clf", SVC(kernel='linear'))  # So that it still is a linear classifier
        ])
    )

    dump(nb_gs.best_estimator_, 'nb_gs.joblib')
    # %% md
    # Gradient Boosted Tree
    # %%
    # Friedman, J.H. (2002). Stochastic gradient boosting. Computational Statistics & Data Analysis, 38, 367-378.
    gbdt_gs = do_grid_search(
        {},
        pipeline=Pipeline([
            vectorizer_factory(),
            ("clf", HistGradientBoostingClassifier())
        ])
    )

    dump(gbdt_gs.best_estimator_, 'gbdt_gs.joblib')