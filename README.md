## Overview

This repository implements an **active learning framework** for training **Sampson**, a neural network emulator for **Sedona**, a radiative transfer simulation code used to model supernova light curves (time series of the light emitted by a star explosion).

Sedona produces high‑fidelity time‑series outputs (e.g., multi‑band light curves) conditioned on explosion and ejecta parameters. However, each Sedona run is expensive, typically requiring **hours** to **days** of wall‑clock time to generate a single time series. Sampson is a learned surrogate model that approximates Sedona’s input–output map, with the goal of reducing evaluation time from hours to **milliseconds** while retaining sufficient fidelity for scientific use.

The core problem this package addresses is: **given a limited budget of Sedona calls, how do we choose simulation points in parameter space to train the emulator most efficiently?** In this package, we use active learning to address this problem.

## What this package does

At a high level, this package:

- Wraps an existing Sampson model and exposes a consistent interface for:
  - Forward prediction on new parameter points.
  - Uncertainty or error proxy evaluation (e.g., via ensembles, MC dropout, or surrogate error metrics).
- Maintains and updates a **training set of (parameters, light curve)** pairs.
- Implements an **active learning loop** that:
  1. Proposes candidate points in the physical parameter space.
  2. Scores these candidates according to a query strategy (e.g., uncertainty, expected error, diversity).
  3. Selects a batch of points to label by running Sedona.
  4. Augments the training set and retrains (or fine‑tunes) Sampson.
- Tracks training/evaluation metrics so we can quantify how emulator accuracy improves as a function of Sedona calls (i.e., simulation budget).

A typical use case is:

1. Start from an initial grid of samples over which to train Sampson.
2. Train an initial Sampson model.
3. Iteratively:
   - Use the current model to propose new high‑value parameter points.
   - Run Sedona at those points (expensive step).
   - Retrain Sampson with the augmented dataset.
4. Stop when emulator accuracy or budget criteria are met.

## Active learning formulation (high level)

Formally, we consider Sedona as an expensive black‑box function
\[
f : \Theta \rightarrow \mathcal{Y},
\]
where \(\Theta\) is the parameter space (e.g., explosion energy, ejecta mass, composition, viewing angle) and \(\mathcal{Y}\) is the space of Sedona outputs (e.g., discretized time‑series fluxes).

Sampson is a parametric model \(\hat{f}_\phi : \Theta \rightarrow \mathcal{Y}\) trained to approximate \(f\). Given a fixed budget \(B\) of Sedona evaluations, we want to choose a sequence of query points
\[
\theta_1, \dots, \theta_B \in \Theta
\]
such that the resulting dataset \(\{(\theta_i, f(\theta_i))\}_{i=1}^B\) yields an emulator with minimal prediction error over a target distribution on \(\Theta\).

This framework supports query strategies of the form:

- **Uncertainty‑based**: prioritize regions where \(\hat{f}_\phi\) has high predictive uncertainty.
- **Error‑based / disagreement‑based**: use ensembles or multiple surrogates to identify regions of high model disagreement.
- **Coverage / diversity‑based**: ensure good coverage of \(\Theta\) to reduce extrapolation.

The concrete strategies implemented in this repository are documented in `docs/active_learning.md` (e.g., acquisition functions, batch selection, and implementation details).