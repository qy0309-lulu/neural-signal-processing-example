# Neural Signal Preprocessing & Analysis Pipeline

Based on the CRCNS HC-3 hippocampal dataset, this project implements a complete pipeline for neural signal preprocessing and analysis. It is a practical implementation of the [learning notes](https://github.com/qy0309-lulu/neural-signal-processing-example/blob/master/Notes.md), covering LFP preprocessing, spike firing rate calculation, behavioral data analysis, and visualization.

## Dataset

Using the **CRCNS HC-3 (Hippocampus)** dataset, which contains neural recordings from rat hippocampus and entorhinal cortex:

| Stat | Count |
|------|-------|
| Total neurons | 7,736 (Pyramidal 6,100 / Interneuron 1,132) |
| Number of sessions | 442 |
| Behavioral task types | 14 |
| Brain regions | EC / CA1 / CA3 / DG |

Data source: [https://crcns.org/data-sets/hc/hc-3](https://crcns.org/data-sets/hc/hc-3)

## Requirements

- Python >= 3.8

## Usage

### Method 1: Jupyter Notebook for visualization

1. `notebook/vis_basis.ipynb`: Basic visualization after data loading (spike raster, time-domain signal plot, PSD plot, trajectory plot)
2. `notebook/LFP_vis.ipynb`: Comparison of LFP time-domain signals, PSD plots, and time-frequency plots before/after preprocessing

### Method 2: Run modules separately

1. `pre_process/load_one_session.py`: Load and display basic information of a session
2. `preprocess_LFP/LFP_prep.py`: LFP preprocessing
3. `preprocess_spike_ch/spike_analysis_ch.py`: Channel-level FR feature extraction from spike signals
4. `preprocess_spike_clu/spike_analysis_clu.py`: Cluster-level FR feature extraction from spike signals

## Project Structure

├─fea_extract
│     cal_speed_whl.py              # calculate speed from .whl file
│     feature_utils_cellType.py     # utility functions
│     feature_utils_LFP.py
│     feature_utils_spike.py
│     fea_extraction_main.py        # extract all features including LFP and spike FR
│  
├─notebook
│     LFP_vis.ipynb                 # visualization of LFP preprocessing
│     vis_basis.ipynb               # basic visualization after loading a session
│
├─preprocess_LFP
│     LFP_analysis.py               # LFP band-specific analysis
│     LFP_prep.py                   # LFP preprocessing code
│     LFP_vis.py                    # code used by LFP_vis.ipynb
│
├─preprocess_spike_ch
│     spike_analysis_ch.py          # spike processing - channel-level FR extraction and analysis
│
├─preprocess_spike_clu
│     spike_analysis_clu.py         # spike processing - cluster-level FR extraction and analysis
│
└─pre_process
       feature_utils_spike.py       # utility functions for spike extraction
       load_one_session.py          # load session data, view basic info (neuron count, channel count, sampling time, etc.)