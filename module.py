
# env_2 has all of these packages installed.

import numpy as np
import pandas as pd

import polars as pl

from scipy import signal as ss   
from scipy.optimize import curve_fit

from scipy.stats import wilcoxon   # Wilcoxon signed-rank test.
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal as kw  #  Kruskal-Wallis H-test for independent samples.
from scipy.stats import normaltest as norm

import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram

# geometric mean & standard deviation.
from scipy.stats import gmean
from scipy.stats import gstd

from scipy.io import savemat
from scipy.io import loadmat

import scipy.interpolate as IPL
from scipy.interpolate import griddata

# %%

import dask.dataframe as dd

from dask.diagnostics import ProgressBar

from dask.distributed import Client
from dask.distributed import get_task_stream

# %%

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

import seaborn as sns
import plotly.express as px

# %%
############

# these 2 modules are not present in env_11 environment.

from open_ephys.analysis import Session

from PyPDF2 import PdfFileMerger, PdfFileReader


import adaptivekde as opt
    # opt
    # Out[39]: <module 'adaptivekde' from '/home/azare/anaconda3/envs/env_2/lib/python3.10/site-packages/adaptivekde/__init__.py'>

################
#
#

# %%

pd.set_option('display.float_format', lambda x: '%.6f' % x)

# sns.set(font_scale=1.75)   #  this makes the default seaborn : gray background with grids.

# sns.set_style("white")
# sns.set(style="whitegrid", font_scale=1.75)

# font_scale : I made it 3 for statistical plots.
sns.set(style="ticks", font_scale=2)  # matplotlib style : only the fontsize is changed.

np.set_printoptions(suppress=True)

# %%
###################

import spikeinterface as si  
import spikeinterface.extractors as se
import spikeinterface.sorters as sst    
import spikeinterface.widgets as sw
import spikeinterface.exporters as exp

from spikeinterface.preprocessing import (bandpass_filter, common_reference)
import spikeinterface.preprocessing as prep 
import spikeinterface.postprocessing as post


import spikeinterface.qualitymetrics as qm
from spikeinterface.qualitymetrics import compute_quality_metrics

# note : this is part of the spikeinterface[full] installation.
# no need to install it separately !!
from probeinterface import read_probeinterface


##############

import spikeinterface.curation as cur

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion_estimation import estimate_motion
from spikeinterface.sortingcomponents.motion_correction import CorrectMotionRecording

#################

import pickle
import copy

# %%

# import hyperspy.api as hs

#
#from math import pi
#
#import sounddevice as sd
#

# %%

from sklearn.pipeline import make_pipeline as mpl
from sklearn.metrics import r2_score , silhouette_score
from sklearn.preprocessing import StandardScaler as stsc
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.linear_model import LinearRegression as LR

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

# %%

import math
import os

import collections

# progress bar
from tqdm import tqdm
tqdm.pandas()  # Enable tqdm for pandas operations
