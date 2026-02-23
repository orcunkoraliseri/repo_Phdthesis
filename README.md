# CENTUS: A Data-Driven Occupancy Modeling Framework

**Journal:** Energy and Buildings
**DOI:** [https://doi.org/10.1016/j.enbuild.2026.117155](https://doi.org/10.1016/j.enbuild.2026.117155)

## Overview

CENTUS is a novel, data-driven methodology for generating high-fidelity residential occupancy data. It synthesizes official population statistics from ISTAT with behavioral patterns analyzed using advanced deep learning architectures (LSTM and Transformer). The framework enables comprehensive, year-long classification of household occupancy across both temporal and non-temporal attributes.

## Highlights

- Presents a realistic occupancy model that integrates population statistics and machine learning
- Classifies all major occupancy attributes simultaneously and annually
- Applicable to all of Italy and HETUS countries
- Provides a holistic annual occupancy classification combining temporal and non-temporal attributes

## Abstract

Occupancy modeling aims to represent the diversity of occupant behavior in buildings. Accurate modeling of occupancy is essential for understanding household dynamics and energy-related interactions in residential buildings, as well as for their use in building performance applications.

The central contribution of this research is CENTUS — a data-driven methodology that synthesizes official population statistics from ISTAT with nuanced behavioral patterns analyzed using LSTM and Transformer architectures. Through unified multitask learning that integrates sequential columns with demographic attributes, these models simultaneously classify multiple occupancy attributes with superior accuracy and broader coverage compared to traditional deterministic and stochastic approaches.

**Key advantages:**
- Privacy protection through ethically sourced public institutional data
- Cross-national compatibility (Italy / HETUS countries)
- Flexible scaling from individual residential units to neighborhood-level analysis via Argmax classification, SoftMax distributions, and temperature-controlled sampling

## Data Sources

| Dataset | Description |
|---|---|
| **ISTAT Census (CensPop2011)** | Italian national population and housing statistics |
| **ISTAT TUS (Time Use Survey 2013)** | Daily and weekly time-use diaries for activity classification |

> Raw dataset files are not included in this repository.

## Repository Structure

```
.
├── datapreprocessing/          # TUS and Census data processing pipelines
│   ├── Census_*.py             # Census housing, occupant, main processing
│   ├── TUS_*.py                # Time Use Survey daily/individual processing
│   ├── 2ndJ_datapreprocessing.py
│   └── preProcessing_Func.py
│
├── architecture/               # Model architecture and training (v3)
│   ├── v3_modeling.py          # LSTM / Transformer model definitions
│   ├── v3_training.py          # Training loop
│   ├── v3_preprocessing.py     # Feature preparation
│   ├── v3_evaluate.py          # Evaluation metrics
│   ├── v3_plotting.py          # Result visualization
│   ├── v3_analysis.py          # Post-training analysis
│   ├── v3_TUS_Data_Feature_Engineering.py
│   ├── v3_mainRNN.py           # RNN entry point
│   ├── v3_mainTrans.py         # Transformer entry point
│   └── v3_mainN_HITS.py        # N-HiTS entry point
│
├── cloud_computing_preparation/ # TRUBA HPC cluster scripts and CENTUS pipelines
│   ├── CENTUS_pipeline_*.py    # Full CENTUS generation pipelines
│   ├── CENTUS_5th_TRUBA_*.py   # TRUBA-optimized LSTM/Transformer runs
│   ├── CENTUS_Data_Equalize_Sequences.py
│   └── old_*/                  # Earlier iteration scripts
│
├── cloud_computing/            # Reserved for future cloud runs
│
├── eSim/                       # EnergyPlus / eSim occupancy integration
│   ├── eSim_datapreprocessing.py
│   ├── eSim_dynamicML_mHead.py
│   ├── eSim_dynamicML_mHead_alignment.py
│   └── eSim_OCC_variations.py
│
├── main.py                     # Main entry point
├── read_stat.py                # Dataset statistics reader
├── impactAnalysis.py           # Occupancy impact analysis
├── EndToEnd_pyTorch_Multitaskv3.py  # End-to-end multitask PyTorch pipeline
└── 00_trial_testaugmentation.py
```

## Methods

The CENTUS framework combines:

- **LSTM** — for sequential occupancy pattern learning from time-use diaries
- **Transformer** — for long-range temporal dependency modeling
- **Multitask learning** — simultaneous classification of multiple occupancy attributes
- **Data synthesis** — merging Census demographics with TUS behavioral profiles

## Requirements

- Python 3.8+
- PyTorch
- TensorFlow / Keras
- CatBoost
- pandas, numpy, scikit-learn
