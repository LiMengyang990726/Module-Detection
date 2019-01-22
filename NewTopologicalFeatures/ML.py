import os
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
from http import HTTPStatus

# from pypi
import matplotlib
import pandas as pd
import requests
import seaborn
import csv
import matplotlib.pyplot as plt
from tabulate import tabulate

topoData = pd.read_csv("allTopoFeatures.csv")

# Do feature scaling
# To make the dataset looks a bit more Gaussian (for the purpose of doing anomaly detection), can do twisting (like take log)
