# <Project title>

MSc Research Project (Cloud Computing), National College of Ireland
Author: Ananya Mohan (25161989) - Supervisor: Dr Giovani Estrada

## What this repository contains
<!-- 2-3 sentences in your own words: what the project does and what is in this repo. -->

## Baseline code used (not included here)
- Smart HPA - https://github.com/HussainAhmad05/Smart_HPA
- ProSmart HPA - https://github.com/HussainAhmad05/ProSmart_HPA
- Online Boutique v0.10.7 - https://github.com/GoogleCloudPlatform/microservices-demo
`prepare_smart_hpa.py` and `prepare_prosmart.py` build runnable copies from these repos and
apply the documented repairs (R1-R4, P1-P12; see each script's header).

## Environment
<!-- AWS Academy Learner Lab, EKS 1.36, 5 x t3.medium, runner t3.large, Ubuntu 24.04.
     Tool versions: see results/env/versions_*.txt -->

## How to run (in this order)
| Step | Script | Purpose |
|---|---|---|
| 1 | (EKS console) | create cluster + node group, runner EC2 |
| 2 | `phase_b.sh` | install Linkerd, viz Prometheus, KEDA, VPA |
| 3 | `run_experiment.sh` | baseline runs: Kubernetes HPA, Smart HPA, ProSmart HPA (2R-20%) |
| 4 | `train_prosmart_models.py` | retrain ProSmart's PROPHET models from Smart HPA runs |
| 5 | `run_mech.sh` (+ `collector.py`, `vertical_actuator.py`, `calibrate_keda.py`, `trace_shape.py`) | training-data runs: HPA / KEDA / vertical x 4 WorldCup98 shapes |
| 6 | `make_labels.py` | label the best mechanism per service and minute |
| 7 | `train_selector.py` | RQ2: train and evaluate the selector |
<!-- add RQ1 forecasting and the final evaluation when done -->

## Data
- `workloads/` - WorldCup98-derived traffic shapes (source: Internet Traffic Archive)
- `results/` - raw results of every run (Locust CSVs, collector metrics, logs, labels)

## Licence
<!-- e.g. MIT for your own code; note the licences of the baseline repositories -->
