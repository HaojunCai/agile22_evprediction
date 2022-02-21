# AGILE 2022: Optimizing Electric Vehicle Charging Schedules Based on Probabilistic Forecast of Individual Mobility
Documentations for reproducible data processing and smart charging simulation and evaluation

## Introduction
Our study predicted the next-day energy consumption and parking duration of electric vehicles (EVs) using three probabilistic models. Furthermore, two time-shifting smart charging strategies were simulated based on the prediction results to evaluate the monetary gains for EV users and peak-shaving effects on distribution grids.

More specifically, all relevant features were first extracted to be fed into prediction models. Secondly, three probabilistic prediction models were trained and evaluated, and feature importances were returned. Thirdly, uncontrolled charging, unidirectional smart charging, and bidirectional smart charging were simulated. Finally, the financial and technical benefits of the two smart charging strategies were evaluated through the comparison with uncontrolled charging.

## Getting Started

In order to run the whole pipeline, you need to run the file main.py. It requires Python 3.

### Prerequisites

The following python packages are required: 
```
* os
* sys
* sqlalchemy
* pandas
* numpy
* datetime
* csv
* scipy
* math
* matplotlib
* geopandas
* trackintel
* skmob
* haversine
* scikit-learn
* statsmodels
* skgarden (! not under good maintenance currently, consider to install directly by git install command)
```

### File Structure
   - main.py: run the whole pipeline
   - extract_mobility.py: extract mobility features
   - extract_evfeatures.py: extract ev-related features 
   - extract_soc.py: extract energy consumption targets, prepare energy consumption inputs for predictions
   - extract_arrival.py: extract arrival time targets, prepare arrival time inputs for predictions
   - extract_depart.py: extract departure time targets, prepare departure time inputs for predictions
   - predict_probablistic_results.py: run three probabilistic models for three targets (energy consumption, arrival time, departure time)
   - calculate_under_overestimation.py: calculate evaluation metrics of three prediction models
   - calculate_feature_importance.py: calculate feature importances
   - compare_probablistic_results.py: compare evaluation metrics among three models
   - evaluate_unidirectional_smartcharging.py: simulate unidirectional smart charging
   - evaluate_bidirectional_smartcharging.py: simulate bidirectional smart charging
   - evaluate_uncontrolled_charging.py: simulate uncontrolled charging as the baseline
   - compare_baseline_unismart.py: compare baseline and unidirectional smart charging
   - compare_baseline_bismart.py: compare baseline and nidirectional smart charging
   - compare_three_charging_onpeakdef2.py: plot load profile of three charging strategies

## Connections
