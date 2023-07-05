
# env_2 has all of these packages installed.

import numpy as np

import pandas as pd

from scipy import signal as ss   
from scipy.optimize import curve_fit

from scipy.stats import wilcoxon
# from scipy.stats import mannwhitneyu

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

############

# these 2 modules are not present in env_11 environment.

from open_ephys.analysis import Session

from PyPDF2 import PdfFileMerger, PdfFileReader

import adaptivekde as opt

################
#
#




###################

import spikeinterface as si  
import spikeinterface.extractors as se
import spikeinterface.sorters as sst    
import spikeinterface.widgets as sw
import spikeinterface.exporters as exp

from spikeinterface.preprocessing import (bandpass_filter, common_reference)
import spikeinterface.postprocessing as post


import spikeinterface.qualitymetrics as qm
from spikeinterface.qualitymetrics import compute_quality_metrics

from probeinterface import read_probeinterface


##############

import spikeinterface.curation as cur

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion_estimation import estimate_motion
from spikeinterface.sortingcomponents.motion_correction import CorrectMotionRecording

#################

import pickle

#

# import hyperspy.api as hs

#
#from math import pi
#
#import sounddevice as sd
#
#
