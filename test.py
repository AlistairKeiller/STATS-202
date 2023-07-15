import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
from ISLP import load_data
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from IPython.display import Latex, Markdown

weekly = load_data("Weekly")
default = load_data("default")
sns.set_theme(style="dark")
