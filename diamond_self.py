import time
import cPickle
import networkx as nx
import numpy as np
import copy
import scipy.stats
from collections import defaultdict
import csv
import sys

G = read_edgelist("gene-network.tsv")
