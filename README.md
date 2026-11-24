# Machine Learning-Based Selection Between Horizontal and Vertical Autoscaling for Kubernetes Microservices

MSc Research Project (Cloud Computing), National College of Ireland
Author: Ananya Mohan (25161989) - Supervisor: Dr Giovani Estrada

## What this repository contains
This project trains a classifier that decides, every minute and for each microservice, whether
the service should scale horizontally (KEDA, more pods) or vertically (VPA recommendations applied
by in-place resize). The selector runs live in a MAPE-K loop on Amazon EKS with Online Boutique and
is compared with HPA, KEDA only, vertical only and a rule-based switcher. The repository holds the
experiment scripts, the base-paper reproduction, the raw data of every run and the full analysis.

## Repository structure
| Folder / file | Contents | Runs on |
|---|---|---|
| `baseline/` | Base-paper reproduction: Kubernetes HPA, Smart HPA and ProSmart HPA runs, repair scripts for the authors' code, retrained ProSmart PROPHET models (`models_2R20/`) | AWS runner |
| `experiments/` | Cluster setup (`phase_b.sh`), one experiment run (`run_mech.sh`), unattended batches (`run_batch.sh`), metrics collector, vertical actuator, live selector, KEDA calibration | AWS runner |
| `load/` | Locust user behaviour (`locust_authors.py`, from Online Boutique) and the WorldCup98 traffic-shape follower (`trace_shape.py`) | AWS runner |
| `workloads/` | `make_workloads.py` and the six traffic shapes built from the WorldCup98 trace | laptop |
| `training/` | RQ2 offline: labels, training table and model comparison, nested tuning, final selector settings, feature-skew check, post-hoc frontend experiment | laptop |
| `analysis/` | Base-paper reproduction summary, RQ1 forecasting comparison and RQ3 evaluation: statistics, cost score, effect sizes, warm-up check, selector behaviour, figures | laptop |
| `notebooks/analysis.ipynb` | Step-by-step walkthrough of data and results | laptop |
| `results/` | Raw data of every run and all derived results (see "Data") | - |
| `reproduce_analysis.py` | Reruns the whole RQ1 + RQ3 analysis from the raw data with one command | laptop |
| `keda_thresholds.json` | Per-service KEDA request-rate targets (from `experiments/calibrate_keda.py`) | runner |

All commands are run from the repository root. Every shell script moves to the root itself,
so it can also be started from another folder.

## Baseline code used (not included here)
- Smart HPA - https://github.com/HussainAhmad05/Smart_HPA
- ProSmart HPA - https://github.com/HussainAhmad05/ProSmart_HPA
- Online Boutique v0.10.7 - https://github.com/GoogleCloudPlatform/microservices-demo (Apache 2.0;
  `load/locust_authors.py` is its load generator)

`baseline/prepare_smart_hpa.py` and `baseline/prepare_prosmart.py` build runnable copies from these
repositories and apply the documented repairs (R1-R4, P1-P12; see each script's header).

## Environment
- AWS Academy Learner Lab, us-east-1. EKS cluster, Kubernetes 1.36, node group of 5 x t3.medium
  (Amazon Linux 2023); runner EC2 t3.large, Ubuntu 24.04, with LabInstanceProfile.
- Online Boutique v0.10.7 (load generator removed), Linkerd edge-26.10.1 with viz, KEDA 2.21.0,
  VPA 1.8.0, Gateway API 1.5.1. Recorded versions: `results/env/`.
- Python: `requirements.txt` (analysis, any computer) and `requirements-runner.txt` (experiments).

## Reproducing the analysis (laptop, no cloud needed)
```
python -m pip install -r requirements.txt
python reproduce_analysis.py
```
This reruns, in order, the base-paper reproduction summary, the RQ3 statistics for both evaluation workloads and the preliminary round,
the post-hoc cost score (three weightings), effect sizes, selector behaviour, the exploratory warm-up
check, the RQ1 forecasting comparison and all figures (about 1.5 minutes). Outputs go to
`results/eval/`, `results/rq1/` and `results/figures/`. Reruns produce identical files.
The post-hoc frontend experiment takes about 10 minutes and is run separately:
`python training/posthoc_frontend.py`.

## Running the experiments (AWS)
Order in which the project was carried out:

| Step | Command | Purpose |
|---|---|---|
| 1 | EKS console | Create the cluster, node group and runner; inbound HTTPS 443 rule from the VPC to the cluster security group |
| 2 | `experiments/phase_b.sh linkerd`, then `viz`, `mesh`, `keda`, `vpa`, `versions` | Install the stack; `experiments/run_mech.sh tune-vpa` makes the VPA recommender react within minutes |
| 3 | `baseline/run_experiment.sh hpa\|smarthpa\|prosmart <run>` | Base-paper reproduction (scenario 2 replicas, 20 % CPU) |
| 4 | `python baseline/train_prosmart_models.py ...` | Retrain ProSmart's PROPHET models from the Smart HPA runs |
| 5 | `python workloads/make_workloads.py` | Build the traffic shapes from WorldCup98 |
| 6 | `experiments/run_mech.sh hpa\|keda\|vertical <shape> <run>` | Training runs: 4 shapes x 3 mechanisms x 2 repeats; `python experiments/calibrate_keda.py` after the first HPA run |
| 7 | `python training/make_labels.py --shape <s> --run <r> --slo-floor 50` | Best mechanism per service and minute |
| 8 | `python training/train_selector.py --root . --floor 50 --runs 1 2 --avg --task binary --lags` | RQ2: training table and model comparison |
| 9 | `python training/tune_selector.py`, then `python training/build_final_model.py` | Nested tuning; fix the final selector (`results/selector/final_selector.json`) |
| 10 | `END_BY=<HH:MM UTC> experiments/run_batch.sh <shape> <first> <last>` | Live evaluation of all five methods; finished runs are skipped, nodes are scaled to 0 at the end |
| 11 | `python reproduce_analysis.py` (laptop) | Analysis and figures |

The final evaluation (eval_heldout runs 11-15, eval_final runs 1-3) ran with the code tagged
`final-eval`, before the repository was reorganised into folders. The scripts' logic is unchanged;
only file locations and the paths between them were updated.

## Data
- `workloads/` - WorldCup98-derived traffic shapes (source: Internet Traffic Archive, Arlitt and Jin, 2000).
  `eval_final` was chosen by a rule fixed before any run (see the header of `workloads/make_workloads.py`).
- `results/mech/` - collector metrics (15-s samples), selector decisions, vertical resize actions per run
- `results/locust/` - Locust statistics per run
- `results/logs/` - batch summary, selector, actuator, collector and Locust logs
- `results/labels/`, `results/selector/` - labels, training tables, RQ2 results, final selector settings
- `results/eval/`, `results/rq1/`, `results/figures/` - analysis outputs
- `results/hpa/`, `results/smarthpa/`, `results/prosmart/`, `results/cluster/` - base-paper reproduction
- `results/invalid/` - runs excluded from the analysis, kept for transparency (pre-fix versions)

## Licence
MIT for the code in this repository. The baseline code (Smart HPA, ProSmart HPA, Online Boutique)
keeps its own licence; see the links under "Baseline code used".
