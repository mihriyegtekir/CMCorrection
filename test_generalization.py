import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import torch
import utils
import models
import pandas as pd
from typing import List, Tuple
import json
import pickle
import gzip
from datetime import datetime
from typing import Optional
import glob
from typing import Dict, Tuple, Set
import evaluate_performance as ep
import inspect
import dcor  # distance correlation library
import shutil
import pyarrow

# Warning: Enabling the configs below (True) while keeping the database-creation configs above disabled (False) can cause the code to error out — necessary update will be applied.

# ===== Config =====
ENABLE_RESULTS_TXT_SKIP = True
ENABLE_FOLDER_SKIP = False
GENERATE_DIAGNOSTIC_PLOTS = False  # validation histograms
ENABLE_FRACTIONAL_PLOTS = False    # fractional RMS plots
RESULTS_FILE = "plots/performance/results.txt"
BASE_DIR = "plots/performance"
ARCH_FILTER = (512, 512, 512, 512, 64)  # only in20__512-512-512-512-64*
APPLY_ARCH_FILTER_TO_AGGREGATES = True  # apply layer filter to ALL existing charts

#Control CM/cov-corr plotting
GENERATE_CM_PLOTS = False

#Create a CM Profile plot for every models include all eval modules
RUN_GLOBAL_CM_PROFILES = False   # Set to False to skip this part entirely
# Exclude specific train_module strings from global_cm_profiles
EXCLUDE_TRAIN_MODULES = [
    "ML_F3W_WXIH0190_ML_F3W_WXIH0191_ML_F3W_WXIH0192_ML_F3W_WXIH0193_ML_F3W_WXIH0194_ML_F3W_WXIH0196_ML_F3W_WXIH0197"
]
MODULES_FOR_GLOBAL_CM_PROFILES = [
    "ML_F3W_WXIH0190",
    "ML_F3W_WXIH0194",
    "ML_F3W_WXIH0197",
    "ML_F3W_WXIH0198"
]  # Only these modules will be processed

# === Noise Fractions Config ===
ENABLE_NOISE_FRACTIONS_PLOT = False         # Enable/disable coherent/incoherent noise plots
NOISE_FRACTIONS_MODULES = ["ML_F3W_WXIH0197"]               # [] → all eval modules, or specify list of module names

# ===================== Global Cov/Cor (concatenate across eval modules) =====================

# --- Config flags ---
ENABLE_GLOBAL_COVCORR = False
GLOBAL_COVCORR_ROOT = os.path.join("plots", "performance", "compare_channels", "global_cov_corr")

GLOBAL_COVCORR_REQUIRE_DR0 = True
GLOBAL_COVCORR_ALL_DR0_MODELS = True

# -------------------------------------------------------------------------------------------
# ===== Analytic bundle config =====
ANALYTIC_ENABLE = False                 # set to False to disable analytic bundle generation
ANALYTIC_OVERWRITE = False             # if an analytic_bundle already exists, skip; if True, overwrite it
ANALYTIC_OUTPUT_SUBDIR = "analytic"    # name of the output subdirectory
ANALYTIC_K_LIST = (0,)     # same logic as evaluate_performance.predict_k()

# Which models/eval modules to process
ANALYTIC_FILTER = {
    "eval_modules": ["ML_F3W_WXIH0190", "ML_F3W_WXIH0191", "ML_F3W_WXIH0192", "ML_F3W_WXIH0193", "ML_F3W_WXIH0194", "ML_F3W_WXIH0196", "ML_F3W_WXIH0197", "ML_F3W_WXIH0198"],     # [] or None -> include all eval modules
    "dropout_allowed": [0, 0.0],                # [] or None -> include all dropout rates
    "nodes_per_layer": [(512, 512, 512, 512, 64)],  # [] or None -> include all architectures
    # optional: restrict by specific training tags
    "train_include_regex": None,             # e.g., r"ML_.*0190.*"
    "train_exclude_regex": None,
}

# ===== Analytic noise fractions (from analytic_bundle) =====
ANALYTIC_NOISE_PLOTS_ENABLE = False          # master switch to run analytic noise plots
ANALYTIC_NOISE_PLOTS_MODULES = [            # [] -> all eval modules; or restrict by names
    "ML_F3W_WXIH0190", "ML_F3W_WXIH0191", "ML_F3W_WXIH0192",
    "ML_F3W_WXIH0193", "ML_F3W_WXIH0194", "ML_F3W_WXIH0196",
    "ML_F3W_WXIH0197", "ML_F3W_WXIH0198",
]
ANALYTIC_NOISE_PLOTS_K_LIST = None

# ===================== Analytic Global Cov/Cor (mirror of DNN path) =====================

ENABLE_ANALYTIC_GLOBAL_COVCORR = False
# Reuse GLOBAL_COVCORR_ROOT from DNN path:
# GLOBAL_COVCORR_ROOT = os.path.join("plots", "performance", "compare_channels", "global_cov_corr")

ANALYTIC_GLOBAL_COVCORR_REQUIRE_DR0 = True
ANALYTIC_GLOBAL_COVCORR_ALL_DR0_MODELS = True  # if False, ta8"] #only the first matching model per train_module

# === Projection hists (from evaluate_performance.py) ===
ENABLE_PROJECTION_HISTS = False
PROJECTION_EVAL_MODULES = ["ML_F3W_WXIH0190", "ML_F3W_WXIH0198"]  # [] -> all modules
PROJECTION_MODEL_REGEX = r"__dr0(\.0)?$"
PROJECTION_TOPK_LIST = [3]   # multiple k values can be executed in one run
SKIP_EXISTING_PROJECTIONS = True  # skip if projection folders already exist

# === Distance corr (from evaluate_performance.py) ===
ENABLE_DISTCORR_PLOTS = False
SKIP_EXISTING_DISTCORR = True
DISTCORR_EVAL_MODULES = ["ML_F3W_WXIH0190", "ML_F3W_WXIH0198"]  # [] -> all modules
DISTCORR_MODEL_REGEX = None

DISTCORR_FILTER = {
    "nodes_per_layer": [(512, 512, 512, 512, 64)],  # [] or None -> all architectures
    "dropout_allowed": [0, 0.0],                    # [] or None -> all dropout rates
}

# ===================== Multiple-Input DNN Train (from global_bundle) =====================
ENABLE_MULTIPLE_INPUT_EXPORT = False
MULTI_INPUT_TAG = "multiple input dnn train"

# Source directories (where global_bundle.pkl.gz files are stored)
MULTI_INPUT_BUNDLE_ROOT = os.path.join("plots", "performance", "compare_channels", "global_cov_corr")

# Which train_module / model combinations to process
MULTI_INPUT_TRAIN_MODULES = [
    "ML_F3W_WXIH0190_ML_F3W_WXIH0191", "ML_F3W_WXIH0190_ML_F3W_WXIH0191_ML_F3W_WXIH0192", "ML_F3W_WXIH0190_ML_F3W_WXIH0191_ML_F3W_WXIH0192_ML_F3W_WXIH0193",
]
MULTI_INPUT_MODELS = [
    "in20__512-512-512-512-64__dr0",
]
# ===================== Multiple-Input DNN Inference (from exported inputs) =====================
ENABLE_MULTI_INPUT_DNN_INFERENCE = False

# Each entry selects a (train_module, model_name) pair whose inputs live under .../inputs
MULTI_DNN_RUNS = [
    {
        "train_module": "ML_F3W_WXIH0190_ML_F3W_WXIH0191",
        "model_name":  "in20__512-512-512-512-64__dr0",
    },
    {
        "train_module": "ML_F3W_WXIH0190_ML_F3W_WXIH0191_ML_F3W_WXIH0192_ML_F3W_WXIH0193",
        "model_name":  "in20__512-512-512-512-64__dr0",
    },
    {
     	"train_module": "ML_F3W_WXIH0190_ML_F3W_WXIH0191_ML_F3W_WXIH0192",
        "model_name":  "in20__512-512-512-512-64__dr0",
    },


]

# Select multiple by index, e.g. [0], [1], [0,1]
ACTIVE_MULTI_DNN_RUNS: List[int] = [0, 1, 2]



# Main output root directory
MULTI_INPUT_OUTPUT_ROOT = os.path.join("plots", "performance", "multiple_module_input")

# ==================== MULTIPLE MODULE INPUT CREATION RUNNER =====================

ENABLE_MULTI_INPUT_FROM_SINGLE_MODULES = False

# Source folders: only read from here
SINGLE_MODULE_INPUT_BASE = "/eos/user/a/areimers/hgcal/dnn_inputs"

# Destination folder: write combined outputs here
MULTI_INPUT_OUTPUT_ROOT = "plots/performance/multiple_module"

# Define which module combinations you want to merge
MULTI_INPUT_RUNS = [
    {
        "modules": ["ML_F3W_WXIH0190", "ML_F3W_WXIH0191"],
        "train_module_name": "ML_F3W_WXIH0190_ML_F3W_WXIH0191",
        "model_name": "in20__512-512-512-512-64__dr0",
    },
    {
        "modules": ["ML_F3W_WXIH0190", "ML_F3W_WXIH0191", "ML_F3W_WXIH0192"],
        "train_module_name": "ML_F3W_WXIH0190_ML_F3W_WXIH0191_ML_F3W_WXIH0192",
        "model_name": "in20__512-512-512-512-64__dr0",
    },
    # --- 4-module combination ---
    {
        "modules": ["ML_F3W_WXIH0190", "ML_F3W_WXIH0191", "ML_F3W_WXIH0192", "ML_F3W_WXIH0193"],
        "train_module_name": "ML_F3W_WXIH0190_ML_F3W_WXIH0191_ML_F3W_WXIH0192_ML_F3W_WXIH0193",
        "model_name": "in20__512-512-512-512-64__dr0",
    },
]

# Which indices from MULTI_INPUT_RUNS to execute
# (0=2 modules, 1=3 modules, 2=4 modules)
ACTIVE_MULTI_INPUT_RUNS_FOR_BUILD = [0, 1, 2]

# === Multiple-Input DNN post-processing (bundle + plots) ===
# Write a predictions bundle under the multiple-module tree (dnn/predictions_bundle.pkl.gz)
ENABLE_MULTI_DNN_BUNDLE = False

# Immediately generate plots after inference, using the bundle saved under dnn/
ENABLE_MULTI_DNN_PROJECTION = False     # projection histograms (nonlinearity test)
ENABLE_MULTI_DNN_DISTCORR  = False     # distance correlation plots

# Projection settings
MULTI_DNN_PROJ_TOPK_LIST = [2]         # generate modes up to k for each value here
SKIP_EXISTING_MULTI_PROJ = True        # skip if the projection folder already has results

# Distance-corr settings
SKIP_EXISTING_MULTI_DIST = True        # skip if the distance folder already has results


# ================== Analytic Results for Multiple Modules( Projection - Distance corr Cov ===========
# --- Configuration for analytic bundle plotting ---
ENABLE_ANALYTIC_PROJECTION = False
ENABLE_ANALYTIC_DISTANCECORR = False
ENABLE_ANALYTIC_DELTA_DISTCORR = False

# Path to the analytic global bundle (example path)
analytic_bundle_path = (
    "plots/performance/compare_channels/global_cov_corr/"
    "ML_F3W_WXIH0190_ML_F3W_WXIH0191_ML_F3W_WXIH0192_ML_F3W_WXIH0193/in20__512-512-512-512-64__dr0/analytic/global_bundle.pkl.gz"
)

# Output directory for new analytic plots
analytic_output_dir = (
    "plots/performance/multiple_module/"
    "ML_F3W_WXIH0190_ML_F3W_WXIH0191_ML_F3W_WXIH0192_ML_F3W_WXIH0193/in20__512-512-512-512-64__dr0/analytic"
)

os.makedirs(analytic_output_dir, exist_ok=True)


# ============================================================
# === CONDITIONAL ANALYTIC BUNDLE LOADING ====================
# ============================================================

if ENABLE_ANALYTIC_PROJECTION or ENABLE_ANALYTIC_DISTANCECORR or ENABLE_ANALYTIC_DELTA_DISTCORR:
    print(f"[analytic] Loading analytic bundle → {analytic_bundle_path}")
    with gzip.open(analytic_bundle_path, "rb") as f:
        bundle = pickle.load(f)

    frames = bundle.get("frames", {})
    true_df = frames.get("true")
    pred_df = frames.get("pred")
    resid_df = frames.get("residual")
    cm_df = frames.get("cm")

    # --- Verify all required frames are available ---
    if true_df is None or pred_df is None or resid_df is None or cm_df is None:
        raise RuntimeError("[analytic] ERROR: Missing one or more frames in analytic bundle")

    print(f"[analytic] Bundle frames loaded:")
    print(f"  true_df:    shape={true_df.shape}")
    print(f"  pred_df:    shape={pred_df.shape}")
    print(f"  residual:   shape={resid_df.shape}")
    print(f"  cm_df:      shape={cm_df.shape}")

    # --- Build variants and residuals dictionaries for plotting ---
    variants = {"true": true_df, "analytic": pred_df}
    residuals = {"analytic": resid_df}

    # --- Create analytic output directories ---
    proj_dir = os.path.join(analytic_output_dir, "projection")
    distcorr_dir = os.path.join(analytic_output_dir, "distance_corr")
    delta_dir = os.path.join(analytic_output_dir, "delta_distance_corr")
    os.makedirs(proj_dir, exist_ok=True)
    os.makedirs(distcorr_dir, exist_ok=True)
    os.makedirs(delta_dir, exist_ok=True)
else:
    print("[analytic] Skipping analytic bundle load (no analytic plots enabled).")
    variants, residuals, cm_df = None, None, None

# ============================================================
# === 1. Projection Plots ====================================
# ============================================================
if ENABLE_ANALYTIC_PROJECTION:
    print(f"[analytic] Generating projection plots → {proj_dir}")
    try:
        ep.plot_all_projection_hists(
            split_name="analytic",
            variants=variants,        # CM-free DataFrames
            residuals=residuals,      # CM-free residuals
            cm_df=cm_df,              # measured CM data
            k=2,                      # top eigenmodes
            plot_dir=proj_dir
        )
        print("[analytic] Projection plots created successfully.")
    except Exception as e:
        print(f"[analytic] WARNING: Projection plot generation failed → {e}")

# ============================================================
# === 2. Distance Corr/Cov + Delta Corr ======================
# ============================================================
if ENABLE_ANALYTIC_DISTANCECORR:
    print(f"[analytic] Generating distance correlation heatmaps → {distcorr_dir}")

    try:
        # Combine CM columns for proper distance correlation calculation
        variants_with_cms = {
            k: ep.add_cms_to_measurements_df(measurements_df=v, cm_df=cm_df, drop_constant_cm=False)
            for k, v in variants.items()
        }
        residuals_with_cms = {
            k: ep.add_cms_to_measurements_df(measurements_df=v, cm_df=cm_df, drop_constant_cm=False)
            for k, v in residuals.items()
        }

        # Dummy minimal config
        class DummyCfg:
            nch_per_erx = 37
        cfg = DummyCfg()

        # Standard distance corr/cov plots
        ep.plot_dist_corr(
            split_name="analytic",
            cfg=cfg,
            variants=variants_with_cms,
            residuals=residuals_with_cms,
            cm_df=cm_df,
            plot_dir=distcorr_dir
        )

        # Delta-linearized distance corr plots
        if ENABLE_ANALYTIC_DELTA_DISTCORR:
            print(f"[analytic] Generating delta distance-corr plots → {delta_dir}")
            ep.plot_delta_lin_dist_corr(
                split_name="analytic",
                cfg=cfg,
                variants=variants_with_cms,
                residuals=residuals_with_cms,
                cm_df=cm_df,
                plot_dir=delta_dir
            )

        print("[analytic] Distance correlation + delta plots created successfully.")
    except Exception as e:
        print(f"[analytic] WARNING: Distance correlation plot generation failed → {e}")

print(f"[analytic] All analytic plots saved to: {analytic_output_dir}")

#===================================================================================

def _p(mod_dir, *xs):
    """Helper to join path parts."""
    return os.path.join(mod_dir, *xs)


def _load_single_module_flat_arrays(mod_dir: str):
    """Load all arrays from one module directory."""
    colnames = []
    try:
        with open(_p(mod_dir, "colnames.json"), "r") as f:
            colnames = json.load(f)
    except Exception as e:
        print(f"[WARN] {mod_dir}: colnames.json could not be read ({e}). Skipped.")

    # Case A: module already has flattened arrays
    if all(os.path.exists(_p(mod_dir, x)) for x in ["inputs.npy", "targets.npy", "eventid.npy", "chadc.npy"]):
        X = np.load(_p(mod_dir, "inputs.npy"))
        y = np.load(_p(mod_dir, "targets.npy")).squeeze()
        eid = np.load(_p(mod_dir, "eventid.npy")).astype(int).squeeze()
        ch = np.load(_p(mod_dir, "chadc.npy")).astype(int).squeeze()
        return X, y, eid, ch, colnames

    # Case B: module has separate train/val files
    needed = [
        "inputs_train.npy", "inputs_val.npy",
        "targets_train.npy", "targets_val.npy",
        "indices_train.npy", "indices_val.npy",
        "eventid.npy", "chadc.npy"
    ]
    if all(os.path.exists(_p(mod_dir, x)) for x in needed):
        X_tr = np.load(_p(mod_dir, "inputs_train.npy"))
        X_va = np.load(_p(mod_dir, "inputs_val.npy"))
        y_tr = np.load(_p(mod_dir, "targets_train.npy")).squeeze()
        y_va = np.load(_p(mod_dir, "targets_val.npy")).squeeze()
        idx_tr = np.load(_p(mod_dir, "indices_train.npy")).astype(int).squeeze()
        idx_va = np.load(_p(mod_dir, "indices_val.npy")).astype(int).squeeze()
        eid_full = np.load(_p(mod_dir, "eventid.npy")).astype(int).squeeze()
        ch_full = np.load(_p(mod_dir, "chadc.npy")).astype(int).squeeze()

        # reconstruct full flattened arrays
        X = np.concatenate([X_tr, X_va], axis=0)
        y = np.concatenate([y_tr, y_va], axis=0)
        eid = np.concatenate([eid_full[idx_tr], eid_full[idx_va]], axis=0)
        ch = np.concatenate([ch_full[idx_tr], ch_full[idx_va]], axis=0)
        return X, y, eid, ch, colnames

    raise FileNotFoundError(f"Missing required files in {mod_dir}.")


def _check_colnames_consistency(colnames_all: list[list[str]]) -> list[str]:
    """Choose the longest available column list as reference."""
    non_empty = [c for c in colnames_all if c]
    if not non_empty:
        return []
    ref = max(non_empty, key=len)
    ref_len = len(ref)
    mism = [(i, len(c)) for i, c in enumerate(colnames_all) if c and c != ref]
    if mism:
        print("⚠ Column name mismatch detected.")
        for i, l in mism:
            print(f"   module index {i}: {l} columns (ref {ref_len})")
    return ref


def _pivot_measurements(values, channels, eventid) -> pd.DataFrame:
    """Build a wide event×channel DataFrame (optional diagnostic)."""
    df = pd.DataFrame({"value": values, "channel": channels.astype(int), "eventid": eventid.astype(int)})
    wide = df.pivot(index="eventid", columns="channel", values="value")
    wide.columns = [f"ch_{c:03d}" for c in wide.columns]
    wide = wide.sort_index().reindex(columns=sorted(wide.columns))
    return wide


def run_build_multi_inputs_from_modules():
    """Main function: merge inputs from multiple modules into one dataset."""
    if not ENABLE_MULTI_INPUT_FROM_SINGLE_MODULES:
        print("[multi-input-build] Disabled.")
        return

    for run_idx in ACTIVE_MULTI_INPUT_RUNS_FOR_BUILD:
        if run_idx < 0 or run_idx >= len(MULTI_INPUT_RUNS):
            print(f"[multi-input-build] Invalid index: {run_idx}")
            continue

        cfg = MULTI_INPUT_RUNS[run_idx]
        modules = cfg["modules"]
        train_module_name = cfg["train_module_name"]
        model_name = cfg["model_name"]

        out_dir = os.path.join(MULTI_INPUT_OUTPUT_ROOT, train_module_name, model_name, "dnn", "inputs")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[multi-input-build] run={run_idx}")
        print(f"  modules: {modules}")
        print(f"  output : {out_dir}")

        inputs_list, targets_list, eid_list, ch_list, colnames_list = [], [], [], [], []
        next_eid_offset = 0

        for m in modules:
            mod_dir = os.path.join(SINGLE_MODULE_INPUT_BASE, m)
            if not os.path.isdir(mod_dir):
                print(f"[multi-input-build] Missing: {mod_dir}")
                continue

            try:
                X, y, eid, ch, colnames = _load_single_module_flat_arrays(mod_dir)
            except Exception as e:
                print(f"[multi-input-build] Error reading {m}: {e}")
                continue

            if not (X.shape[0] == y.shape[0] == eid.shape[0] == ch.shape[0]):
                raise RuntimeError(f"[{m}] inconsistent row counts.")

            # apply eventid offset to keep them unique across modules
            eid_shifted = eid + next_eid_offset
            next_eid_offset = int(eid_shifted.max()) + 1

            inputs_list.append(X)
            targets_list.append(y)
            eid_list.append(eid_shifted)
            ch_list.append(ch)
            colnames_list.append(list(colnames) if colnames else [])

        if not inputs_list:
            print("[multi-input-build] No modules loaded, skipped.")
            continue

        colnames = _check_colnames_consistency(colnames_list)

        inputs_merged = np.concatenate(inputs_list, axis=0)
        targets_merged = np.concatenate(targets_list, axis=0)
        eventid_merged = np.concatenate(eid_list, axis=0)
        chadc_merged = np.concatenate(ch_list, axis=0)

        print(f"  merged: inputs={inputs_merged.shape}, targets={targets_merged.shape}")

        np.save(os.path.join(out_dir, "inputs_flat.npy"), inputs_merged)
        np.save(os.path.join(out_dir, "targets_flat.npy"), targets_merged)
        np.save(os.path.join(out_dir, "eventid.npy"), eventid_merged.astype(np.int64))
        np.save(os.path.join(out_dir, "chadc.npy"), chadc_merged.astype(np.int32))
        with open(os.path.join(out_dir, "colnames.json"), "w") as f:
            json.dump(colnames, f, indent=2)

        try:
            meas_df = _pivot_measurements(targets_merged, chadc_merged, eventid_merged)
            meas_df.to_parquet(os.path.join(out_dir, "measurements.parquet"))
        except Exception as e:
            print(f"  measurement parquet not written: {e}")

        with open(os.path.join(out_dir, "log.txt"), "w") as f:
            f.write("===== Multi-Input Merge =====\n")
            f.write(f"Timestamp     : {datetime.utcnow().isoformat()}Z\n")
            f.write(f"Modules       : {modules}\n")
            f.write(f"Train module  : {train_module_name}\n")
            f.write(f"Model name    : {model_name}\n")
            f.write(f"Inputs shape  : {inputs_merged.shape}\n")
            f.write(f"Targets shape : {targets_merged.shape}\n")
            f.write(f"Unique events : {len(np.unique(eventid_merged))}\n")

        print(f"  written to: {out_dir}")
        print(f"[multi-input-build] done: {train_module_name} | {model_name}")
# ===================== /Multiple-Input: build inputs from single-module dirs =====================



# ===================== DNN RUNNER for MULTIPLE MODULES=============================
def run_multi_input_dnn_inference(username_load_model_from: str = "areimers"):

    if not ENABLE_MULTI_INPUT_DNN_INFERENCE:
        print("[multi-dnn] Skipped (ENABLE_MULTI_INPUT_DNN_INFERENCE=False).")
        return

    # where inputs/ and outputs/ live (your area)
    base_io_root = MULTI_INPUT_OUTPUT_ROOT  # "plots/performance/multiple_module_input"
    # where models live (Andreas' area)
    dnn_models_base = f"/eos/user/{username_load_model_from[0]}/{username_load_model_from}/hgcal/dnn_models"

    for idx in ACTIVE_MULTI_DNN_RUNS:
        if idx < 0 or idx >= len(MULTI_DNN_RUNS):
            print(f"[multi-dnn] Invalid index in ACTIVE_MULTI_DNN_RUNS: {idx} (skip)")
            continue

        cfg = MULTI_DNN_RUNS[idx]
        train_module = cfg["train_module"]
        model_name   = cfg["model_name"]

        # ---- build paths ----
        in_dir  = os.path.join(base_io_root, _sanitize(train_module), _sanitize(model_name), "dnn", "inputs")
        out_dir = os.path.join(base_io_root, _sanitize(train_module), _sanitize(model_name), "dnn", "outputs")
        os.makedirs(out_dir, exist_ok=True)

        model_path = os.path.join(
            dnn_models_base, train_module, model_name, "regression_dnn_best.pth"
        )

        print(f"\n[multi-dnn] Run index={idx}")
        print(f"  inputs   : {in_dir}")
        print(f"  model    : {model_path}")
        print(f"  outputs  : {out_dir}")

        # ---- load inputs ----
        try:
            inputs_flat = np.load(os.path.join(in_dir, "inputs_flat.npy"))
            targets_flat = np.load(os.path.join(in_dir, "targets_flat.npy"))
            eventid_flat = np.load(os.path.join(in_dir, "eventid.npy"))
            chadc_flat   = np.load(os.path.join(in_dir, "chadc.npy"))
            meas_df      = pd.read_parquet(os.path.join(in_dir, "measurements.parquet"))
        except Exception as e:
            print(f"[multi-dnn] ERROR loading inputs from {in_dir}: {e}")
            continue

        # ---- instantiate model skeleton (use nodes & dropout parsed from model_name) ----
        nodes_per_layer, dropout_rate = parse_model_config(model_name)
        if nodes_per_layer is None or dropout_rate is None:
            print(f"[multi-dnn] Cannot parse model_name: {model_name} (skip)")
            continue

        input_dim = int(inputs_flat.shape[1])
        device = torch.device("cpu")

        net = models.DNNFlex(
            input_dim=input_dim,
            nodes_per_layer=nodes_per_layer,
            dropout_rate=dropout_rate,
            tag=""
        ).to(device).eval()

        if not os.path.exists(model_path):
            print(f"[multi-dnn] MISSING weights: {model_path} (skip)")
            continue

        # ---- load weights & run inference ----
        try:
            state = torch.load(model_path, map_location=device)
            net.load_state_dict(state)
            # ---- safe inference in chunks to avoid OOM ----
            DNN_INFER_CHUNK_SIZE = 1_000_000  # number of samples per chunk

            preds_list = []
            n_samples = inputs_flat.shape[0]
            with torch.no_grad():
                for start in range(0, n_samples, DNN_INFER_CHUNK_SIZE):
                    end = min(start + DNN_INFER_CHUNK_SIZE, n_samples)
                    X_chunk = torch.from_numpy(inputs_flat[start:end]).float().to(device)
                    y_chunk = net(X_chunk).squeeze().cpu().numpy()
                    preds_list.append(y_chunk)
                    print(f"[multi-dnn] processed {end}/{n_samples} samples")

            y_pred = np.concatenate(preds_list, axis=0)

        except Exception as e:
            print(f"[multi-dnn] ERROR during inference: {e}")
            continue

        # ---- validate sizes & save flat preds ----
        if y_pred.shape[0] != targets_flat.shape[0]:
            raise RuntimeError(
                f"[multi-dnn] Size mismatch: preds={y_pred.shape[0]}, targets={targets_flat.shape[0]}"
            )

        flat_out = os.path.join(out_dir, "preds_flat.npy")
        np.save(flat_out, y_pred)
        print(f"[multi-dnn] Saved flat predictions → {flat_out}")

        # ---- pivot to (event x channel) using evaluate_performance helper ----
        try:
            preds_df = ep.pivot_flat_preds_to_event_channel(
                preds_flat=y_pred,
                eventid_flat=eventid_flat,
                channels_flat=chadc_flat,
                reference_meas_df=meas_df,   # keeps exact event set/column order
            )
        except Exception as e:
            print(f"[multi-dnn] ERROR pivoting predictions: {e}")
            continue

        df_out = os.path.join(out_dir, "preds_df.parquet")
        preds_df.to_parquet(df_out)
        print(f"[multi-dnn] Saved pivoted predictions → {df_out}")

        # ===============================================================
        # Add compatibility block for evaluate_performance layout
        # ===============================================================
        # Standard path: plots/performance/<EVAL>/<TRAIN>/<MODEL>/
        std_dir = os.path.join(
            "plots", "performance",
            _sanitize(train_module),   # <EVAL>
            _sanitize(train_module),   # <TRAIN>
            _sanitize(model_name)      # <MODEL>
        )
        os.makedirs(std_dir, exist_ok=True)

        # 1) Save minimal artifacts for evaluate_performance compatibility
        np.save(os.path.join(std_dir, "predictions_flat.npy"), y_pred)
        preds_df.to_parquet(os.path.join(std_dir, "predictions_df.parquet"))
        try:
            meas_df.to_parquet(os.path.join(std_dir, "measurements.parquet"))
        except Exception as e:
            print(f"[multi-dnn/std] WARN: could not write measurements.parquet ({e})")

        try:
            shutil.copy(os.path.join(in_dir, "eventid.npy"), os.path.join(std_dir, "eventid.npy"))
            shutil.copy(os.path.join(in_dir, "chadc.npy"),   os.path.join(std_dir, "chadc.npy"))
        except Exception as e:
            print(f"[multi-dnn/std] WARN: could not copy id arrays ({e})")

        # 2) Rebuild CM dataframe from inputs + colnames
        try:
            with open(os.path.join(in_dir, "colnames.json")) as f:
                colnames_inputs = json.load(f)
        except Exception:
            colnames_inputs = [f"cm_erx{i:02d}" for i in range(min(12, inputs_flat.shape[1]))]

        ncm = len([c for c in colnames_inputs if str(c).startswith("cm_erx")])
        try:
            _inputs_df, cm_df = _build_input_and_cm_df(
                inputs_flat=inputs_flat,
                eventid_flat=eventid_flat,
                ncm=ncm,
                colnames_inputs=colnames_inputs
            )
            cm_df.to_parquet(os.path.join(std_dir, "cm.parquet"))
        except Exception as e:
            print(f"[multi-dnn/std] WARN: could not reconstruct cm_df ({e})")
            cm_df = pd.DataFrame(index=meas_df.index)

        # 3) Compute residuals and cov/corr with CM columns included
        residual_df = meas_df - preds_df
        df_true_for_cov = add_cms_to_measurements_df(measurements_df=meas_df,  cm_df=cm_df, drop_constant_cm=False)
        df_pred_for_cov = add_cms_to_measurements_df(measurements_df=preds_df, cm_df=cm_df, drop_constant_cm=False)
        df_res_for_cov  = add_cms_to_measurements_df(measurements_df=residual_df, cm_df=cm_df, drop_constant_cm=False)

        cov_true, corr_true = compute_cov_corr(df_true_for_cov)
        cov_pred, corr_pred = compute_cov_corr(df_pred_for_cov)
        cov_res,  corr_res  = compute_cov_corr(df_res_for_cov)

        nch_per_erx = 37
        nerx = max(1, preds_df.shape[1] // nch_per_erx)

        # 4) Save a complete predictions bundle at standard path
        save_predictions_bundle(
            plotfolder=std_dir,
            eval_module=train_module,
            train_module=train_module,
            model_name=model_name,
            nodes_per_layer=list(nodes_per_layer),
            dropout_rate=float(dropout_rate),
            ncmchannels=int(ncm),
            nch_per_erx=int(nch_per_erx),
            nerx=int(nerx),
            eventid_combined=eventid_flat,
            chadc_combined=chadc_flat,
            colnames_inputs=colnames_inputs,
            meas_true_df=meas_df,
            meas_pred_df=preds_df,
            cm_df=cm_df,
            residual_df=residual_df,
            cov_true=cov_true,  corr_true=corr_true,
            cov_pred=cov_pred,  corr_pred=corr_pred,
            cov_res=cov_res,    corr_res=corr_res,
        )

        print(f"[multi-dnn] Also saved standard files and bundle → {std_dir}")
        # ===============================================================

        # ---- small log ----
        with open(os.path.join(out_dir, "log.txt"), "w") as f:
            f.write("===== Multiple-Input DNN Inference =====\n")
            f.write(f"Timestamp     : {datetime.utcnow().isoformat()}Z\n")
            f.write(f"Train module  : {train_module}\n")
            f.write(f"Model name    : {model_name}\n")
            f.write(f"Model path    : {model_path}\n")
            f.write(f"Inputs shape  : {inputs_flat.shape}\n")
            f.write(f"Targets shape : {targets_flat.shape}\n")
            f.write(f"Preds shape   : {y_pred.shape}\n")
            f.write(f"Output folder : {out_dir}\n")

        print(f"[multi-dnn] DONE for {train_module} | {model_name}")


def run_multi_dnn_postprocess_only():
    """
    Recreate a predictions_bundle.pkl.gz and generate projection + distance plots
    from existing multiple-module outputs (no inference).
    """
    # base path where your multiple-module input/output lives
    base_io_root = MULTI_INPUT_OUTPUT_ROOT  # "plots/performance/multiple_module"

    for cfg in MULTI_DNN_RUNS:
        train_module = cfg["train_module"]
        model_name   = cfg["model_name"]

        print(f"\n[postprocess] Starting bundle + plots for {train_module} | {model_name}")

        dnn_dir = os.path.join(base_io_root, _sanitize(train_module), _sanitize(model_name), "dnn")
        input_dir  = os.path.join(dnn_dir, "inputs")
        output_dir = os.path.join(dnn_dir, "outputs")

        # --- expected files ---
        preds_flat_path = os.path.join(output_dir, "preds_flat.npy")
        meas_path        = os.path.join(input_dir, "measurements.parquet")
        eventid_path     = os.path.join(input_dir, "eventid.npy")
        chadc_path       = os.path.join(input_dir, "chadc.npy")

        if not all(os.path.exists(p) for p in [preds_flat_path, meas_path, eventid_path, chadc_path]):
            print(f"[postprocess] Missing required files in {input_dir} or {output_dir}. Skipping.")
            continue

        # --- load everything ---
        preds_flat  = np.load(preds_flat_path)
        eventid_flat = np.load(eventid_path)
        chadc_flat   = np.load(chadc_path)
        meas_df      = pd.read_parquet(meas_path)
        
        # --- DEBUG ---
        #cm_cols_in_meas = [c for c in meas_df.columns if str(c).startswith("cm_erx")]
        #print("[DEBUG] meas_df cm_erx cols:", cm_cols_in_meas)
        #print("[DEBUG] meas_df cm_erx count:", len(cm_cols_in_meas))

        # pivot flat preds to (event × channel)
        preds_df = ep.pivot_flat_preds_to_event_channel(
            preds_flat=preds_flat,
            eventid_flat=eventid_flat,
            channels_flat=chadc_flat,
            reference_meas_df=meas_df
        )

        # --- construct bundle path ---
        bundle_path = os.path.join(dnn_dir, "predictions_bundle.pkl.gz")

        if ENABLE_MULTI_DNN_BUNDLE:
            print(f"[postprocess] Writing bundle → {bundle_path}")

            # 1) (legacy) Try to get CM from measurements.parquet (only if such columns exist)
            cm_cols = [c for c in meas_df.columns if str(c).startswith("cm_erx")]
            if len(cm_cols) > 0:
                cm_df = meas_df[cm_cols]
                print(f"[postprocess] CM taken from measurements.parquet with {len(cm_cols)} columns.")
            else:
                # 2) Otherwise, rebuild CM from inputs_flat.npy using colnames.json
                # --- Try to rebuild CM dataframe from inputs_flat.npy ---
                colnames_path = os.path.join(input_dir, "colnames.json")
                if os.path.exists(colnames_path):
                    with open(colnames_path, "r") as f:
                        colnames_inputs = json.load(f)
                else:
                    print(f"[postprocess] WARNING: colnames.json not found in {input_dir}, using empty list")
                    colnames_inputs = []

                cm_names = [c for c in colnames_inputs if str(c).startswith("cm_erx")]
                try:
                    # --- Load inputs and event IDs ---
                    inputs_arr   = np.load(os.path.join(input_dir, "inputs_flat.npy"), mmap_mode="r")
                    eventid_flat = np.load(os.path.join(input_dir, "eventid.npy"))

                    # --- Identify one row per event (first occurrence) ---
                    unique_ids, first_pos = np.unique(eventid_flat, return_index=True)
                    order = np.argsort(unique_ids)
                    rows = first_pos[order]
                    evt_idx = unique_ids[order]

                    # --- Extract CM columns by name ---
                    if cm_names:
                        cm_indices = [colnames_inputs.index(name) for name in cm_names]
                        cm_data = inputs_arr[rows[:, None], cm_indices]
                        cm_df = pd.DataFrame(cm_data, index=evt_idx, columns=cm_names)
                        print(f"[postprocess] CM rebuilt from inputs_flat.npy with {len(cm_names)} columns.")
                    else:
                        cm_df = pd.DataFrame(index=meas_df.index)
                        print("[postprocess] No CM columns found in colnames.json; using empty cm_df.")

                    # --- Compute residuals and correlation matrices ---
                    print("[postprocess] Computing residuals and correlation matrices...")
                    residual_df = meas_df - preds_df

                    df_true_for_cov = ep.add_cms_to_measurements_df(meas_df, cm_df, drop_constant_cm=False)
                    df_pred_for_cov = ep.add_cms_to_measurements_df(preds_df, cm_df, drop_constant_cm=False)
                    df_res_for_cov  = ep.add_cms_to_measurements_df(residual_df, cm_df, drop_constant_cm=False)

                    cov_true, corr_true = ep.compute_cov_corr(df_true_for_cov)
                    cov_pred, corr_pred = ep.compute_cov_corr(df_pred_for_cov)
                    cov_res,  corr_res  = ep.compute_cov_corr(df_res_for_cov)

                    # --- Save bundle including CM and residuals ---
                    save_predictions_bundle(
                        plotfolder=dnn_dir,
                        eval_module=train_module,
                        train_module=train_module,
                        model_name=model_name,
                        nodes_per_layer=[512,512,512,512,64],
                        dropout_rate=0.0,
                        ncmchannels=len(cm_names),
                        nch_per_erx=37,
                        nerx=6,
                        eventid_combined=eventid_flat,
                        chadc_combined=chadc_flat,
                        colnames_inputs=colnames_inputs,
                        meas_true_df=meas_df,
                        meas_pred_df=preds_df,
                        cm_df=cm_df,
                        residual_df=residual_df,
                        cov_true=cov_true,  corr_true=corr_true,
                        cov_pred=cov_pred,  corr_pred=corr_pred,
                        cov_res=cov_res,    corr_res=corr_res,
                    )

                except Exception as e:
                    print(f"[postprocess] WARNING: failed to rebuild CM or save bundle → {e}")
                    cm_df = pd.DataFrame(index=meas_df.index)



        # --- generate projection + distance plots from bundle ---
        if ENABLE_MULTI_DNN_PROJECTION:
            proj_dir = os.path.join(dnn_dir, "plots", "projection")
            os.makedirs(proj_dir, exist_ok=True)
            for k in MULTI_DNN_PROJ_TOPK_LIST:
                print(f"[postprocess] Projection (k={k}) → {proj_dir}")
                _run_projection_hists_via_ep(
                    bundle_path=bundle_path,
                    out_dir=proj_dir,
                    title=f"{train_module} | {model_name}",
                    nbins=60,
                    k=k
                )

        if ENABLE_MULTI_DNN_DISTCORR:
            dist_dir = os.path.join(dnn_dir, "plots", "distance")
            os.makedirs(dist_dir, exist_ok=True)
            print(f"[postprocess] Distance corr → {dist_dir}")
            _run_distcorr_via_ep(
                bundle_path=bundle_path,
                out_dir=dist_dir,
                title=f"{train_module} | {model_name}"
            )

        print(f"[postprocess] Done for {train_module} | {model_name}")



def _sanitize(s: str) -> str:
    """Replace unsafe characters for file paths."""
    return s.replace("/", "_").replace(" ", "_")

"""
def _export_from_global_bundle(bundle_path: str, out_dir_inputs: str) -> None:
    os.makedirs(out_dir_inputs, exist_ok=True)

    # --- open bundle ---
    with gzip.open(bundle_path, "rb") as f:
        b = pickle.load(f)

    frames = b.get("frames", {}) or {}
    true_df: pd.DataFrame = frames.get("true")
    cm_df:   pd.DataFrame = frames.get("cm")

    if true_df is None or cm_df is None:
        raise RuntimeError(f"Missing 'true' or 'cm' in bundle: {bundle_path}")

    # --- sort channel columns (ch_000, ch_001, ...) ---
    ch_cols = [c for c in true_df.columns if str(c).startswith("ch_")]
    if not ch_cols:
        raise RuntimeError("No channel columns starting with 'ch_' found in 'true' frame.")
    ch_cols_sorted = sorted(ch_cols, key=lambda c: int(c.split("_")[1]))
    true_df = true_df[ch_cols_sorted]

    # --- dimensions ---
    n_events = len(true_df)
    n_channels = len(ch_cols_sorted)
    n_cm = cm_df.shape[1]

    # --- build flattened arrays ---
    # inputs_flat: repeat CM vector of each event for all channels
    inputs_flat = np.repeat(cm_df.to_numpy(), n_channels, axis=0)          # (E*C, N_cm)
    # targets_flat: flatten true values row-wise
    targets_flat = true_df.to_numpy().ravel(order="C")                     # (E*C,)
    # eventid and channel arrays
    event_ids = true_df.index.to_numpy()
    eventid_flat = np.repeat(event_ids, n_channels)                         # (E*C,)
    ch_nums = np.array([int(c.split("_")[1]) for c in ch_cols_sorted], dtype=int)
    chadc_flat = np.tile(ch_nums, n_events)                                 # (E*C,)

    # --- input column names ---
    if all(str(c).startswith("cm_erx") for c in cm_df.columns):
        colnames_inputs = list(map(str, cm_df.columns))
    else:
        colnames_inputs = [f"cm_{i:02d}" for i in range(n_cm)]

    # --- write outputs ---
    np.save(os.path.join(out_dir_inputs, "inputs_flat.npy"),   inputs_flat)
    np.save(os.path.join(out_dir_inputs, "targets_flat.npy"),  targets_flat)
    np.save(os.path.join(out_dir_inputs, "eventid.npy"),       eventid_flat)
    np.save(os.path.join(out_dir_inputs, "chadc.npy"),         chadc_flat)

    true_df.to_parquet(os.path.join(out_dir_inputs, "measurements.parquet"))
    cm_df.to_parquet(os.path.join(out_dir_inputs, "cm.parquet"))

    with open(os.path.join(out_dir_inputs, "colnames.json"), "w") as f:
        json.dump(colnames_inputs, f)

    # --- summary log ---
    print(f"[multi-input] {bundle_path}")
    print(f"  events={n_events}  channels={n_channels}  N_cm={n_cm}")
    print(f"  → wrote to {out_dir_inputs}")

def run_multiple_input_export():
    if not ENABLE_MULTIPLE_INPUT_EXPORT:
        print("[multi-input] Skipped (ENABLE_MULTIPLE_INPUT_EXPORT=False).")
        return

    total, done = 0, 0
    for train_module in MULTI_INPUT_TRAIN_MODULES:
        for model_name in MULTI_INPUT_MODELS:
            total += 1
            bundle_dir = os.path.join(MULTI_INPUT_BUNDLE_ROOT, train_module, model_name)
            bundle_path = os.path.join(bundle_dir, "global_bundle.pkl.gz")
            if not os.path.exists(bundle_path):
                print(f"[multi-input] MISSING bundle: {bundle_path}")
                continue

            # output directory:
            # plots/performance/multiple_module_input/<TRAIN_MODULE>/<MODEL>/inputs
            out_dir_inputs = os.path.join(
                MULTI_INPUT_OUTPUT_ROOT,
                _sanitize(train_module),
                _sanitize(model_name),
                "inputs"
            )
            try:
                _export_from_global_bundle(bundle_path=bundle_path, out_dir_inputs=out_dir_inputs)
                done += 1
            except Exception as e:
                print(f"[multi-input] ERROR for {train_module} / {model_name}: {e}")

    print(f"[multi-input] Finished: {done}/{total} exports.")
# ===================== /Multiple-Input DNN Train (from global_bundle) =====================
"""

# ================== Distance Correlation Runner ===================

def _has_valid_results(folder: str) -> bool:
    """
    Check if the folder contains valid output files (PDF/PNG).
    Avoids skipping empty folders.
    """
    if not os.path.exists(folder):
        return False
    for f in os.listdir(folder):
        if f.endswith(".pdf") or f.endswith(".png"):
            return True
        subpath = os.path.join(folder, f)
        if os.path.isdir(subpath):
            # check recursively inside subfolders (e.g., distcorr/, delta_lin_distcorr/)
            if _has_valid_results(subpath):
                return True
    return False

def _run_distcorr_via_ep(bundle_path: str, out_dir: str, title: str = ""):
    """
    Load a prediction or analytic bundle and generate distance correlation plots.
    Reads the pickled bundle, extracts DataFrames, builds a minimal EvalConfig,
    and calls ep.plot_dist_corr() and ep.plot_delta_lin_dist_corr().
    """
    import gzip, pickle, re
    os.makedirs(out_dir, exist_ok=True)

    # --- Load bundle file ---
    with gzip.open(bundle_path, "rb") as f:
        b = pickle.load(f)

    frames = b.get("frames", {}) or {}
    true_df = frames.get("true")
    pred_df = frames.get("pred")
    resid_df = frames.get("residual")
    cm_df = frames.get("cm")

    # --- Handle analytic bundles separately ---
    if "analytic" in bundle_path:
        if pred_df is None and "analytic" in frames:
            if isinstance(frames["analytic"], dict) and 0 in frames["analytic"]:
                pred_df = frames["analytic"][0]
        if resid_df is None and "residuals" in frames:
            if isinstance(frames["residuals"], dict) and 0 in frames["residuals"]:
                resid_df = frames["residuals"][0]

    # --- Sanity checks ---
    if true_df is None or pred_df is None:
        raise RuntimeError("Both 'true' and 'pred' frames are required for distance correlation plots.")
    if cm_df is None or cm_df.empty:
        raise RuntimeError("'cm_df' is required for distance correlation plots but was not found in the bundle.")


    # --- Build a minimal EvalConfig to satisfy ep.plot_dist_corr() ---
    # Some plotting functions inside evaluate_performance expect cfg.nch_per_erx, cfg.nerx, cfg.ncmchannels, etc.
    # We create a minimal placeholder config with safe default values.
    try:
        module_match = re.search(r"(ML_[A-Z0-9_]+)", bundle_path)
        eval_mod = module_match.group(1) if module_match else "unknown"
    except Exception:
        eval_mod = "unknown"

    cfg = ep.EvalConfig(
        modulenames_used_for_training=[eval_mod],
        modulename_for_evaluation=eval_mod,
        nodes_per_layer=(512, 512, 512, 512, 64),
        dropout_rate=0.0,
        modeltag="distance_corr",
        inputfoldertag="auto",
        ncmchannels=12,   # number of common-mode channels per erx
        nch_per_erx=37,   # number of physical channels per erx
        nerx=6,           # number of erx chips (or sensor partitions)
    )

    # --- Prepare variants and residuals dictionaries ---
    variants = {"true": true_df, "dnn": pred_df}
    residuals = {"dnn": resid_df} if resid_df is not None else {}

    # This ensures that CM channels are included in distance correlation plots,
    # avoiding axis misalignment and missing CM blocks.
    variants_with_cms = {
        k: ep.add_cms_to_measurements_df(
            measurements_df=v,
            cm_df=cm_df,
            drop_constant_cm=False
        )
        for k, v in variants.items()
    }

    residuals_with_cms = {
        k: ep.add_cms_to_measurements_df(
            measurements_df=v,
            cm_df=cm_df,
            drop_constant_cm=False
        )
        for k, v in residuals.items()
    }

    # --- Generate distance correlation plots ---
    print(f"[distcorr] Generating plots in {out_dir}")

    ep.plot_dist_corr(
        split_name="combined",
        cfg=cfg,
        variants=variants_with_cms,       # <-- changed here
        residuals=residuals_with_cms,     # <-- changed here
        cm_df=cm_df,
        plot_dir=os.path.join(out_dir, "distcorr"),
    )

    ep.plot_delta_lin_dist_corr(
        split_name="combined",
        cfg=cfg,
        variants=variants_with_cms,       # <-- changed here
        residuals=residuals_with_cms,     # <-- changed here
        cm_df=cm_df,
        plot_dir=os.path.join(out_dir, "delta_lin_distcorr"),
    )


def run_distance_corr_plots():
    """
    Iterate over all bundles and generate distance correlation plots
    (both DNN and Analytic). Uses skip logic and configurable module filters.
    """
    if not ENABLE_DISTCORR_PLOTS:
        print("[distcorr] Skipped (ENABLE_DISTCORR_PLOTS=False).")
        return

    import re as _re
    patt = _re.compile(DISTCORR_MODEL_REGEX) if DISTCORR_MODEL_REGEX else None
    total, done = 0, 0

    for eval_module, train_module, model_name, bundle_path in _iter_all_bundles(BASE_DIR):
        nodes, _dr = parse_model_config(model_name)

        # --- Config-based architecture/dropout filtering ---
        allowed_layers = DISTCORR_FILTER.get("nodes_per_layer", [])
        if allowed_layers and nodes not in allowed_layers:
            continue

        allowed_drops = DISTCORR_FILTER.get("dropout_allowed", [])
        if allowed_drops and _dr not in allowed_drops:
            continue

        if DISTCORR_EVAL_MODULES and (eval_module not in DISTCORR_EVAL_MODULES):
            continue
        if patt and (patt.search(model_name) is None):
            continue

        total += 1
        print(f"\n[distcorr] Processing model={model_name} (layers={nodes}, dr={_dr})")

        # -------- DNN --------
        out_dir_dnn = os.path.join(BASE_DIR, eval_module, train_module, model_name, "distance_corr", "dnn")
        if SKIP_EXISTING_DISTCORR and _has_valid_results(out_dir_dnn):
            print(f"[distcorr][DNN] Skipped existing results: {out_dir_dnn}")
        else:
            try:
                _run_distcorr_via_ep(bundle_path=bundle_path, out_dir=out_dir_dnn, title=f"{model_name} [DNN]")
                print(f"[distcorr][DNN] Done → {out_dir_dnn}")
                done += 1
            except Exception as e:
                print(f"[distcorr][DNN] ERROR ({model_name}): {e}")

        # -------- ANALYTIC --------
        analytic_bundle = os.path.join(BASE_DIR, eval_module, train_module, model_name, "analytic", "analytic_bundle.pkl.gz")
        out_dir_analytic = os.path.join(BASE_DIR, eval_module, train_module, model_name, "distance_corr", "analytic")

        if os.path.exists(analytic_bundle):
            if SKIP_EXISTING_DISTCORR and _has_valid_results(out_dir_analytic):
                print(f"[distcorr][ANALYTIC] Skipped existing results: {out_dir_analytic}")
            else:
                try:
                    _run_distcorr_via_ep(bundle_path=analytic_bundle, out_dir=out_dir_analytic, title=f"{model_name} [Analytic]")
                    print(f"[distcorr][ANALYTIC] Done → {out_dir_analytic}")
                    done += 1
                except Exception as e:
                    print(f"[distcorr][ANALYTIC] ERROR ({model_name}): {e}")
        else:
            print(f"[distcorr][ANALYTIC] Skipped (no analytic bundle found): {analytic_bundle}")

    print(f"\n[distcorr] Finished: {done}/{total} model bundles processed.")

# ================== Projection Histogram - nonlinearity Test ===========
def _run_projection_hists_via_ep(
    bundle_path: str,
    out_dir: str,
    title: str = "",
    nbins: int = 60,
    k: int = None,
):
    """
    Load a prediction or analytic bundle and generate projection histograms.
    Accepts a user-defined 'k' value (from PROJECTION_TOPK_LIST).
    """
    with gzip.open(bundle_path, "rb") as f:
        b = pickle.load(f)
    """
    # --- DEBUG: inspect bundle content ---
    print(f"[DEBUG] Inspecting bundle: {bundle_path}")
    print(b.keys())
    if "frames" in b:
        print("Frames keys:", list(b["frames"].keys()))
    else:
        print("No 'frames' key in bundle.")
    # -------------------------------------
    """
    frames = b.get("frames", {}) or {}
    true_df = frames.get("true")
    pred_df = frames.get("pred")
    resid_df = frames.get("residual")
    cm_df = frames.get("cm")

    # --- only apply this logic for ANALYTIC bundles ---
    if "analytic" in bundle_path:
        if pred_df is None and "analytic" in frames:
            if isinstance(frames["analytic"], dict) and 0 in frames["analytic"]:
                pred_df = frames["analytic"][0]
                print("[DEBUG] (Analytic) Using frames['analytic'][0] as prediction source.")

        if resid_df is None and "residuals" in frames:
            if isinstance(frames["residuals"], dict) and 0 in frames["residuals"]:
                resid_df = frames["residuals"][0]
                print("[DEBUG] (Analytic) Using frames['residuals'][0] as residual source.")

    if true_df is None or pred_df is None:
        raise RuntimeError("Both 'true' and 'pred' frames are required for projection.")
    ch_cols = [c for c in true_df.columns if str(c).startswith("ch_")]
    if not ch_cols:
        raise RuntimeError("No ch_* columns found for projection.")
    if len(true_df) != len(pred_df):
        nmin = min(len(true_df), len(pred_df))
        true_df = true_df.iloc[:nmin].reset_index(drop=True)
        pred_df = pred_df.iloc[:nmin].reset_index(drop=True)
        if resid_df is not None:
            resid_df = resid_df.iloc[:nmin].reset_index(drop=True)

    if cm_df is None or cm_df.empty:
        raise RuntimeError("The 'cm_df' is required for projection but was not found in the bundle.")

    variants = {"true": true_df, "dnn": pred_df}
    residuals = {}
    if resid_df is not None:
        residuals["dnn"] = resid_df

    # Detect valid arguments from ep.plot_all_projection_hists
    sig = inspect.signature(ep.plot_all_projection_hists)
    params = set(sig.parameters.keys())

    kwargs = {}
    if "variants" in params:
        kwargs["variants"] = variants
    if "residuals" in params:
        kwargs["residuals"] = residuals
    if "split_name" in params:
        kwargs["split_name"] = "combined"
    if "cm_df" in params:
        kwargs["cm_df"] = cm_df
    if "plot_dir" in params:
        kwargs["plot_dir"] = out_dir
    if "k" in params and k is not None:
        # plot_all_projection_hists draws modes 0..k-1
        # calling with k+1 ensures mode k is included
        kwargs["k"] = k + 1

    print(f"[projection] Generating projections up to mode {k} in {out_dir}")


    # Directly generate all projection histograms into a single folder
    os.makedirs(out_dir, exist_ok=True)
    ep.plot_all_projection_hists(**kwargs)

    print(f"[projection] Completed: all modes up to {k} stored in {out_dir}")


    print(f"[projection] Calling plot_all_projection_hists with k={k}")
    return ep.plot_all_projection_hists(**kwargs)


def run_projection_hists():
    """
    Iterate over existing DNN and Analytic bundles and generate projection histograms
    for multiple k values. Skip existing results if enabled.
    """
    if not ENABLE_PROJECTION_HISTS:
        print("[projection] Skipped (ENABLE_PROJECTION_HISTS=False).")
        return

    import re as _re
    patt = _re.compile(PROJECTION_MODEL_REGEX) if PROJECTION_MODEL_REGEX else None
    total, done = 0, 0

    for eval_module, train_module, model_name, bundle_path in _iter_all_bundles(BASE_DIR):
        nodes, _dr = parse_model_config(model_name)
        if ARCH_FILTER and (nodes != ARCH_FILTER):
            continue
        if PROJECTION_EVAL_MODULES and (eval_module not in PROJECTION_EVAL_MODULES):
            continue
        if patt and (patt.search(model_name) is None):
            continue

        total += 1

        # Path to potential analytic bundle
        analytic_bundle = os.path.join(
            BASE_DIR, eval_module, train_module, model_name, "analytic", "analytic_bundle.pkl.gz"
        )

        # Loop over all selected k values
        for k_value in PROJECTION_TOPK_LIST:
            print(f"\n[projection] Processing model={model_name}, k={k_value}")

            # ======== DNN ========
            out_dir_dnn = os.path.join(
                BASE_DIR, eval_module, train_module, model_name, "projections"
            )

            if SKIP_EXISTING_PROJECTIONS and os.path.exists(out_dir_dnn) and os.listdir(out_dir_dnn):
                print(f"[projection][DNN] Skipped existing results for k={k_value}: {out_dir_dnn}")
            else:
                os.makedirs(out_dir_dnn, exist_ok=True)
                try:
                    print(f"[projection][DNN] Generating histograms for k={k_value}")
                    _run_projection_hists_via_ep(
                        bundle_path=bundle_path,
                        out_dir=out_dir_dnn,
                        title=f"{eval_module} | {train_module} | {model_name} [DNN k={k_value}]",
                        nbins=60,
                        k=k_value,
                    )
                    print(f"[projection][DNN] Wrote histograms to: {out_dir_dnn}")
                    done += 1
                except Exception as e:
                    print(f"[projection][DNN] ERROR ({eval_module}, {train_module}, {model_name}, k={k_value}): {e}")

            # ======== ANALYTIC ========
            if os.path.exists(analytic_bundle):
                out_dir_analytic = os.path.join(
                    BASE_DIR, eval_module, train_module, model_name, "analytic", "projections"
                )

                if SKIP_EXISTING_PROJECTIONS and os.path.exists(out_dir_analytic) and os.listdir(out_dir_analytic):
                    print(f"[projection][ANALYTIC] Skipped existing results for k={k_value}: {out_dir_analytic}")
                else:
                    os.makedirs(out_dir_analytic, exist_ok=True)
                    try:
                        print(f"[projection][ANALYTIC] Generating histograms for k={k_value}")
                        _run_projection_hists_via_ep(
                            bundle_path=analytic_bundle,
                            out_dir=out_dir_analytic,
                            title=f"{eval_module} | {train_module} | {model_name} [Analytic k={k_value}]",
                            nbins=60,
                            k=k_value,
                        )
                        print(f"[projection][ANALYTIC] Wrote histograms to: {out_dir_analytic}")
                        done += 1
                    except Exception as e:
                        print(f"[projection][ANALYTIC] ERROR ({eval_module}, {train_module}, {model_name}, k={k_value}): {e}")
            else:
                print(f"[projection][ANALYTIC] Skipped (no analytic bundle found): {analytic_bundle}")

    print(f"\n[projection] Finished: {done}/{total} model bundles processed.")


# =================== /Projection histograms runner =====================



def _iter_analytic_bundles_for(train_module: str, model_name: str):
    """
    Yield (eval_module, bundle_path) for analytic bundles under:
      plots/performance/<EVAL>/<TRAIN>/<MODEL>/analytic/analytic_bundle.pkl.gz
    Eval list is parsed from train_module exactly like DNN path.
    """
    eval_modules = _get_eval_modules_from_train_module(train_module)
    if not eval_modules:
        print(f"[agcc-simple] WARN: no evals parsed from '{train_module}'")
        return
    for emod in eval_modules:
        p = os.path.join(
            BASE_DIR, emod, train_module, model_name, "analytic", "analytic_bundle.pkl.gz"
        )
        if os.path.exists(p):
            yield emod, p
        else:
            print(f"[agcc-simple] MISSING: {p}")

def _load_frames_from_analytic_bundle(bundle_path: str, k: int = 0):
    """
    Return (true_df, pred_df, cm_df, meta) from an analytic bundle.
    Use frames['analytic'][k] if present; otherwise reconstruct pred = true - residuals[k].
    """
    with gzip.open(bundle_path, "rb") as f:
        b = pickle.load(f)
    frames = b.get("frames", {}) or {}
    meta   = b.get("meta",   {}) or {}

    true_df = frames.get("true")
    cm_df   = frames.get("cm")
    anal    = frames.get("analytic", {}) or {}
    resid   = frames.get("residuals", {}) or {}
    if true_df is None:
        raise RuntimeError(f"'true' frame missing in {bundle_path}")
    if k in anal:
        pred_df = anal[k]
    elif k in resid:
        pred_df = true_df - resid[k]
    else:
        raise RuntimeError(f"Neither analytic[k={k}] nor residuals[k={k}] in {bundle_path}")
    return true_df, pred_df, cm_df, meta

def _build_global_covcorr_analytic(train_module: str, model_name: str, k: int = 0):
    """
    Mirror of _build_global_covcorr_for but reading analytic bundles and
    writing outputs under .../<train_module>/<model_name>/analytic/.
    Uses the same _save_global_bundle_and_plots() helper for identical outputs.
    """
    bundles = list(_iter_analytic_bundles_for(train_module, model_name))
    if not bundles:
        print(f"[agcc-simple] SKIP: no analytic bundles for {train_module} / {model_name}")
        return

    trues, preds, cms = [], [], []
    nch_per_erx_guess = None
    per_eval_counts = []

    for emod, bpath in bundles:
        try:
            true_df, pred_df, cm_df, meta_b = _load_frames_from_analytic_bundle(bpath, k=k)
        except Exception as e:
            print(f"[agcc-simple] WARN: cannot open {bpath}: {e}")
            continue

        # Keep only channel columns (ch_*)
        ch_cols = [c for c in true_df.columns if str(c).startswith("ch_")]
        if ch_cols:
            true_df = true_df[ch_cols]
            pred_df = pred_df[ch_cols]

        trues.append(true_df)
        preds.append(pred_df)
        cms.append(cm_df if cm_df is not None else pd.DataFrame(index=true_df.index))
        per_eval_counts.append((emod, len(true_df)))

        if nch_per_erx_guess is None:
            try:
                nch_per_erx_guess = int(meta_b.get("nch_per_erx", 37))
            except Exception:
                nch_per_erx_guess = 37

    if not trues:
        print(f"[agcc-simple] SKIP: no readable frames for {train_module} / {model_name}")
        return

    # Concatenate rows (events) across eval modules; union columns if needed
    true_global = _align_columns_or_union(trues, kind="TRUE").reset_index(drop=True)
    pred_global = _align_columns_or_union(preds, kind="PRED").reset_index(drop=True)
    cm_global   = _align_columns_or_union(cms,   kind="CM").reset_index(drop=True)

    # Trim to common length if needed
    nmin = min(len(true_global), len(pred_global), len(cm_global) if len(cm_global) else len(true_global))
    if not (len(true_global) == len(pred_global) == nmin):
        print(f"[agcc-simple] WARN: row mismatch after concat; trimming to {nmin}")
        true_global = true_global.iloc[:nmin].reset_index(drop=True)
        pred_global = pred_global.iloc[:nmin].reset_index(drop=True)
        if len(cm_global):
            cm_global   = cm_global.iloc[:nmin].reset_index(drop=True)

    residual_global = true_global - pred_global

    # Info
    print(f"[agcc-simple] Modules in '{train_module}':")
    for em, n in per_eval_counts:
        print(f"  - {em}: {n} events")
    print(f"[agcc-simple] Global concatenated shape for {train_module}/{model_name}: "
          f"{len(true_global)} events, {true_global.shape[1]} channels")

    # Compute cov/corr with CM columns included (no dropping constant CM; same as DNN global)
    df_true = add_cms_to_measurements_df(true_global, cm_global, drop_constant_cm=False)
    df_pred = add_cms_to_measurements_df(pred_global, cm_global, drop_constant_cm=False)
    df_res  = add_cms_to_measurements_df(residual_global, cm_global, drop_constant_cm=False)

    cov_true, corr_true = compute_cov_corr(df_true)
    cov_pred, corr_pred = compute_cov_corr(df_pred)
    cov_res,  corr_res  = compute_cov_corr(df_res)

    # Output dir: identical tree as DNN + 'analytic' leaf
    out_dir = os.path.join(GLOBAL_COVCORR_ROOT, _sanitize(train_module), _sanitize(model_name), "analytic")
    _save_global_bundle_and_plots(
        out_dir, train_module, model_name,
        cov_true, corr_true, cov_pred, corr_pred, cov_res, corr_res,
        true_global, pred_global, residual_global, cm_global,
        nch_per_erx_guess or 37,
    )
    # Note: uses same filenames as DNN (e.g., global_bundle.pkl.gz, corr_true.pdf, ...)

def run_analytic_global_covcorr():
    """
    Mirror of run_global_covcorr for analytic inputs.
    - Discovers train_modules and models under dnn_models_base
    - Filters by dr0 and ARCH (same as DNN)
    - For each selected model, builds analytic global cov/corr (k=0)
    """
    if not ENABLE_ANALYTIC_GLOBAL_COVCORR:
        print("[agcc-simple] Skipped (ENABLE_ANALYTIC_GLOBAL_COVCORR=False).")
        return

    username_load_model_from = "areimers"
    dnn_models_base = f"/eos/user/{username_load_model_from[0]}/{username_load_model_from}/hgcal/dnn_models"

    for train_module in sorted(os.listdir(dnn_models_base)):
        train_dir = os.path.join(dnn_models_base, train_module)
        if not os.path.isdir(train_dir):
            continue

        model_names = sorted([
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ])
        if not model_names:
            continue

        # Filter on dropout/architecture (same as DNN path)
        filtered = []
        for mname in model_names:
            nodes_per_layer, dr = parse_model_config(mname)
            if ANALYTIC_GLOBAL_COVCORR_REQUIRE_DR0 and (dr is None or abs(dr - 0) > 1e-12):
                continue
            if ARCH_FILTER and (nodes_per_layer != ARCH_FILTER):
                continue
            filtered.append(mname)

        if not filtered:
            print(f"[agcc-simple] NOTE: no valid models under {train_module}")
            continue

        selected = filtered if ANALYTIC_GLOBAL_COVCORR_ALL_DR0_MODELS else filtered[:1]
        for model_name in selected:
            print(f"[agcc-simple] Building analytic global cov/corr for {train_module} / {model_name}")
            _build_global_covcorr_analytic(train_module, model_name, k=0)
# ================= END of ANALYTIC GLOBAL CORR COV ======================================

def _get_eval_modules_from_train_module(train_module: str) -> list:
    """
    Parse eval modules from train_module string.
    Example:
        'ML_F3W_WXIH0190_ML_F3W_WXIH0191' ->
        ['ML_F3W_WXIH0190', 'ML_F3W_WXIH0191']
    """
    if not isinstance(train_module, str) or "ML_" not in train_module:
        return []

    parts = train_module.split("_ML_")
    tokens = []
    for i, p in enumerate(parts):
        if i == 0:
            tok = p if p.startswith("ML_") else ("ML_" + p)
        else:
            tok = "ML_" + p
        tokens.append(tok)

    print(f"[global-covcorr] Parsed eval modules from '{train_module}': {tokens}")
    return tokens

def _iter_bundles_for(train_module: str, model_name: str):
    """
    Yield (eval_module, bundle_path, meta) for each eval module.
    Bundle path is always:
        BASE_DIR/<EVAL>/<TRAIN>/<MODEL>/predictions_bundle.pkl.gz
    """
    eval_modules = _get_eval_modules_from_train_module(train_module)
    if not eval_modules:
        print(f"[global-covcorr] WARN: Could not parse any eval modules from: {train_module}")
        return

    found_any = False
    for emod in eval_modules:
        bundle_path = os.path.join(
            BASE_DIR, emod, train_module, model_name, "predictions_bundle.pkl.gz"
        )
        if not os.path.exists(bundle_path):
            # Helpful diagnostics: show what folders do exist under BASE_DIR/<EVAL>/<TRAIN>
            probe_dir = os.path.join(BASE_DIR, emod, train_module)
            print(f"[global-covcorr] MISSING: {bundle_path}")
            if not os.path.exists(probe_dir):
                print(f"[global-covcorr] NOTE: Directory does not exist: {probe_dir}")
                # List what we DO have for this eval module to help spot typos
                probe_parent = os.path.join(BASE_DIR, emod)
                if os.path.isdir(probe_parent):
                    try:
                        kids = sorted(os.listdir(probe_parent))
                        print(f"[global-covcorr] Existing under {probe_parent}: {kids[:10]}{' ...' if len(kids)>10 else ''}")
                    except Exception:
                        pass
            continue

        try:
            with gzip.open(bundle_path, "rb") as f:
                b = pickle.load(f)
            meta = b.get("meta", {})
            found_any = True
            print(f"[global-covcorr] FOUND bundle for eval='{emod}': {bundle_path}")
            yield emod, bundle_path, meta
        except Exception as e:
            print(f"[global-covcorr] WARN: cannot open {bundle_path}: {e}")

    if not found_any:
        print(f"[global-covcorr] SKIP: no bundles for any eval under train='{train_module}', model='{model_name}'")


def _load_frames_from_bundle(bundle_path: str):
    """
    Load frames and meta from a predictions bundle.
    Returns (true_df, pred_df, cm_df, meta)
    """
    with gzip.open(bundle_path, "rb") as f:
        b = pickle.load(f)
    frames = b.get("frames", {})
    meta   = b.get("meta", {})
    return frames.get("true"), frames.get("pred"), frames.get("cm"), meta


def _align_columns_or_union(dfs: list, kind: str) -> pd.DataFrame:
    """
    Row-wise concatenate DataFrames from different eval modules as if they were new events.
    If columns differ, take the union and reindex. Always reset/ignore the index.
    """
    if not dfs:
        return pd.DataFrame()
    col_sets = [tuple(df.columns) for df in dfs]
    if all(cols == col_sets[0] for cols in col_sets):
        return pd.concat(dfs, axis=0, ignore_index=True)
    union_cols = sorted(set().union(*[df.columns for df in dfs]))
    print(f"[global-covcorr] INFO: {kind} columns differ; using union of {len(union_cols)} cols.")
    dfs_re = [df.reindex(columns=union_cols) for df in dfs]
    return pd.concat(dfs_re, axis=0, ignore_index=True)


class _SimpleSplit:
    """Minimal split object with only what build_variants() needs."""
    def __init__(self, measurements_df: pd.DataFrame, cm_df: pd.DataFrame, event_ids: np.ndarray):
        self.measurements_df = measurements_df
        self.cm_df = cm_df
        self.event_ids = event_ids

def _build_train_pool_split(modnames: List[str], base_cfg: ep.EvalConfig) -> _SimpleSplit:
    """
    Build the *fit* domain from multiple training modules by row-wise concatenation.
    Columns are aligned by UNION (missing columns are filled with NaN).
    Event IDs are re-numbered to be contiguous from 0..N-1.
    """
    meas_list, cm_list, eid_list = [], [], []

    for m in modnames:
        # clone base config but point evaluation module to current train module 'm'
        cfg_i = ep.EvalConfig(
            modulenames_used_for_training=base_cfg.modulenames_used_for_training,
            modulename_for_evaluation=m,                 # <--- this is the *source* of rows
            nodes_per_layer=base_cfg.nodes_per_layer,
            dropout_rate=base_cfg.dropout_rate,
            modeltag=base_cfg.modeltag,
            inputfoldertag=base_cfg.inputfoldertag,
            ncmchannels=base_cfg.ncmchannels,
            nch_per_erx=base_cfg.nch_per_erx,
            nerx=base_cfg.nerx,
        )
        io_i = ep.DataIO(cfg=cfg_i)
        io_i.load_all()
        s_i = io_i.get_split("combined")  # same split semantics you already use

        meas_list.append(s_i.measurements_df)
        cm_list.append(s_i.cm_df)
        eid_list.append(np.asarray(s_i.event_ids, dtype=int))

    if not meas_list:
        raise RuntimeError("No training modules found to pool for analytic fit.")

    # union columns across modules, then row-wise concat (events stacked one under another)
    meas_pool = _align_columns_or_union(meas_list, kind="MEAS_POOL")
    cm_pool   = _align_columns_or_union(cm_list,   kind="CM_POOL")

    # rebuild contiguous event ids for the pooled table
    nrows = len(meas_pool)
    eids  = np.arange(nrows, dtype=int)

    # safety: keep only ch_* in measurements (same convention as elsewhere)
    ch_cols = [c for c in meas_pool.columns if str(c).startswith("ch_")]
    if ch_cols:
        meas_pool = meas_pool[ch_cols]

    return _SimpleSplit(measurements_df=meas_pool, cm_df=cm_pool, event_ids=eids)


def _save_global_bundle_and_plots(
    out_dir: str,
    train_module: str,
    model_name: str,
    cov_true: pd.DataFrame, corr_true: pd.DataFrame,
    cov_pred: pd.DataFrame, corr_pred: pd.DataFrame,
    cov_res:  pd.DataFrame, corr_res:  pd.DataFrame,
    true_df: pd.DataFrame, pred_df: pd.DataFrame,
    residual_df: pd.DataFrame, cm_df: pd.DataFrame,
    nch_per_erx: int
):
    os.makedirs(out_dir, exist_ok=True)

    # Info file
    with open(os.path.join(out_dir, "info.txt"), "w") as f:
        f.write(f"train_module: {train_module}\n")
        f.write(f"model_name: {model_name}\n")
        f.write(f"events_total: {true_df.shape[0]}\n")
        f.write(f"channels: {true_df.shape[1]}\n")

    bundle = {
        "version": 1,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "meta": {
            "scope": "global_covcorr",
            "train_module": train_module,
            "model_name": model_name,
            "nch_per_erx": int(nch_per_erx),
        },
        "frames": {
            "true": true_df,
            "pred": pred_df,
            "residual": residual_df,
            "cm": cm_df,
        },
        "covcorr": {
            "true": {"cov": cov_true, "corr": corr_true},
            "pred": {"cov": cov_pred, "corr": corr_pred},
            "residual": {"cov": cov_res, "corr": corr_res},
        },
    }
    out_pkl = os.path.join(out_dir, "global_bundle.pkl.gz")
    with gzip.open(out_pkl, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[global-covcorr] wrote {out_pkl}")

    # Plots
    plot_covariance(corr_true, nch_per_erx, "Correlation (global true)",
                    "channel i", "channel j", "corr(i,j)",
                    os.path.join(out_dir, "corr_true.pdf"), zrange=(-1., 1.))
    plot_covariance(corr_pred, nch_per_erx, "Correlation (global prediction)",
                    "channel i", "channel j", "corr(i,j)",
                    os.path.join(out_dir, "corr_pred.pdf"), zrange=(-1., 1.))
    plot_covariance(corr_res, nch_per_erx, "Correlation (global residual)",
                    "channel i", "channel j", "corr(i,j)",
                    os.path.join(out_dir, "corr_residual.pdf"), zrange=(-1., 1.))

    plot_covariance(cov_true, nch_per_erx, "Covariance (global true)",
                    "channel i", "channel j", "cov(i,j)",
                    os.path.join(out_dir, "cov_true.pdf"), zrange=(-4., 4.))
    plot_covariance(cov_pred, nch_per_erx, "Covariance (global prediction)",
                    "channel i", "channel j", "cov(i,j)",
                    os.path.join(out_dir, "cov_pred.pdf"), zrange=(-4., 4.))
    plot_covariance(cov_res, nch_per_erx, "Covariance (global residual)",
                    "channel i", "channel j", "cov(i,j)",
                    os.path.join(out_dir, "cov_residual.pdf"), zrange=(-4., 4.))


def _build_global_covcorr_for(train_module: str, model_name: str):
    """
    For a given train_module/model_name: collect all eval bundles, stack rows as new events,
    then compute global cov/corr.
    """
    bundles = list(_iter_bundles_for(train_module, model_name))
    if not bundles:
        print(f"[global-covcorr] SKIP: no bundles for {train_module} / {model_name}")
        return

    trues, preds, cms = [], [], []
    nch_per_erx_guess = None
    per_eval_counts = []

    for emod, bpath, meta in bundles:
        true_df, pred_df, cm_df, meta_b = _load_frames_from_bundle(bpath)
        if true_df is None or pred_df is None:
            print(f"[global-covcorr] WARN: frames missing in {bpath}")
            continue

        # Keep only channel columns (ch_*)
        ch_cols = [c for c in true_df.columns if str(c).startswith("ch_")]
        if ch_cols:
            true_df = true_df[ch_cols]
            pred_df = pred_df[ch_cols]

        trues.append(true_df)
        preds.append(pred_df)
        cms.append(cm_df if cm_df is not None else pd.DataFrame(index=true_df.index))
        per_eval_counts.append((emod, len(true_df)))

        if nch_per_erx_guess is None:
            try:
                nch_per_erx_guess = int(meta_b.get("nch_per_erx", 37))
            except Exception:
                nch_per_erx_guess = 37

    if not trues:
        print(f"[global-covcorr] SKIP: no readable frames for {train_module} / {model_name}")
        return

    # Row-wise concat as "new events" and fully reset the index to ensure contiguous event IDs
    true_global = _align_columns_or_union(trues, kind="TRUE").reset_index(drop=True)
    pred_global = _align_columns_or_union(preds, kind="PRED").reset_index(drop=True)
    cm_global   = _align_columns_or_union(cms,   kind="CM").reset_index(drop=True)

    # Safety: ensure row counts match (trim if necessary)
    nmin = min(len(true_global), len(pred_global), len(cm_global) if len(cm_global) else len(true_global))
    if not (len(true_global) == len(pred_global) == nmin):
        print(f"[global-covcorr] WARN: row mismatch after concat; trimming to {nmin}")
        true_global = true_global.iloc[:nmin].reset_index(drop=True)
        pred_global = pred_global.iloc[:nmin].reset_index(drop=True)
        if len(cm_global):
            cm_global   = cm_global.iloc[:nmin].reset_index(drop=True)

    residual_global = true_global - pred_global

    # Informative logging
    print(f"[global-covcorr] Modules in '{train_module}':")
    for em, n in per_eval_counts:
        print(f"  - {em}: {n} events")
    print(f"[global-covcorr] Global concatenated shape for {train_module}/{model_name}: "
          f"{len(true_global)} events, {true_global.shape[1]} channels")

    # Include CM columns when building matrices for cov/corr
    df_true_for_cov = add_cms_to_measurements_df(true_global, cm_global, drop_constant_cm=False)
    df_pred_for_cov = add_cms_to_measurements_df(pred_global, cm_global, drop_constant_cm=False)
    df_res_for_cov  = add_cms_to_measurements_df(residual_global, cm_global, drop_constant_cm=False)

    cov_true, corr_true = compute_cov_corr(df_true_for_cov)
    cov_pred, corr_pred = compute_cov_corr(df_pred_for_cov)
    cov_res,  corr_res  = compute_cov_corr(df_res_for_cov)

    out_dir = os.path.join(GLOBAL_COVCORR_ROOT, _sanitize(train_module), _sanitize(model_name))
    _save_global_bundle_and_plots(
        out_dir, train_module, model_name,
        cov_true, corr_true, cov_pred, corr_pred, cov_res, corr_res,
        true_global, pred_global, residual_global, cm_global,
        nch_per_erx_guess or 37,
    )


def run_global_covcorr():
    """
    Iterate train modules (only to discover which models exist),
    then for each (train_module, model_name) read bundles from plots/performance/<EVAL>/<TRAIN>/<MODEL>.
    """
    if not ENABLE_GLOBAL_COVCORR:
        print("[global-covcorr] Skipped (ENABLE_GLOBAL_COVCORR=False).")
        return

    os.makedirs(GLOBAL_COVCORR_ROOT, exist_ok=True)

    username_load_model_from = "areimers"
    dnn_models_base = f"/eos/user/{username_load_model_from[0]}/{username_load_model_from}/hgcal/dnn_models"

    for train_module in sorted(os.listdir(dnn_models_base)):
        train_dir = os.path.join(dnn_models_base, train_module)
        if not os.path.isdir(train_dir):
            continue

        model_names = sorted([
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ])
        if not model_names:
            continue

        # Filter on dropout/architecture as requested
        filtered = []
        for mname in model_names:
            nodes_per_layer, dr = parse_model_config(mname)
            if GLOBAL_COVCORR_REQUIRE_DR0 and (dr is None or abs(dr - 0) > 1e-12):
                continue
            if ARCH_FILTER and (nodes_per_layer != ARCH_FILTER):
                continue
            filtered.append(mname)

        if not filtered:
            print(f"[global-covcorr] NOTE: no valid models under {train_module}")
            continue

        selected = filtered if GLOBAL_COVCORR_ALL_DR0_MODELS else filtered[:1]
        for model_name in selected:
            print(f"[global-covcorr] Building global cov/corr for {train_module} / {model_name}")
            _build_global_covcorr_for(train_module, model_name)

# ===================== /Global Cov/Cor block =====================


def _arch_ok(layer_tuple):
    """Return True if row passes ARCH filter for aggregates."""
    return (not APPLY_ARCH_FILTER_TO_AGGREGATES) or (layer_tuple == ARCH_FILTER)


existing_result_keys = set()

def _sanitize(name: str) -> str:
    """Filesystem-safe: keep letters, numbers, _ and -; collapse others to '_'."""
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', name).strip('_')

def _iter_all_bundles(base="plots/performance"):
    """
    Yields (eval_module, train_module, model_name, bundle_path).
    """
    pattern = os.path.join(base, "*", "*", "*", "predictions_bundle.pkl.gz")
    for p in glob.glob(pattern):
        # .../plots/performance/<EVAL>/<TRAIN>/<MODEL>/predictions_bundle.pkl.gz
        parts = p.split(os.sep)
        eval_module, train_module, model_name = parts[-4], parts[-3], parts[-2]
        yield eval_module, train_module, model_name, p

def _open_bundle(bundle_path):
    with gzip.open(bundle_path, "rb") as f:
        b = pickle.load(f)
    meta = b["meta"]
    corr_true = b["covcorr"]["true"]["corr"]
    corr_pred = b["covcorr"]["pred"]["corr"]
    return meta, corr_true, corr_pred

def _open_bundle_full(bundle_path):
    """Return (meta, meas_true_df, corr_true, corr_pred) from a bundle."""
    with gzip.open(bundle_path, "rb") as f:
        b = pickle.load(f)
    meta = b["meta"]
    # event×channel TRUE table saved in bundle
    meas_true_df = b["frames"]["true"]
    corr_true = b["covcorr"]["true"]["corr"]
    corr_pred = b["covcorr"]["pred"]["corr"]
    return meta, meas_true_df, corr_true, corr_pred


def plot_covcorr_from_bundle(bundle_path: str,
                             out_dir: Optional[str] = None,
                             zcorr=(-1., 1.),
                             zcov=(-4., 4.)) -> None:
    with gzip.open(bundle_path, "rb") as f:
        b = pickle.load(f)

    meta    = b["meta"]
    cc      = b["covcorr"]
    nch_per = int(meta["nch_per_erx"])
    if out_dir is None:
        out_dir = os.path.dirname(bundle_path)
    os.makedirs(out_dir, exist_ok=True)

    cov_true,  corr_true  = cc["true"]["cov"],     cc["true"]["corr"]
    cov_pred,  corr_pred  = cc["pred"]["cov"],     cc["pred"]["corr"]
    cov_res,   corr_res   = cc["residual"]["cov"], cc["residual"]["corr"]

    def _p(name): return os.path.join(out_dir, f"{name}.pdf")

    # Correlation
    utils.plot_covariance(df=corr_true, nch_per_erx=nch_per, title="Correlation (true)",
                          xtitle="channel i", ytitle="channel j", ztitle="corr(i,j)",
                          zrange=zcorr, output_filename=_p("correlation_matrix_true"))
    utils.plot_covariance(df=corr_pred, nch_per_erx=nch_per, title="Correlation (prediction)",
                          xtitle="channel i", ytitle="channel j", ztitle="corr(i,j)",
                          zrange=zcorr, output_filename=_p("correlation_matrix_pred"))
    utils.plot_covariance(df=corr_res, nch_per_erx=nch_per, title="Correlation (residuals)",
                          xtitle="channel i", ytitle="channel j", ztitle="corr(i,j)",
                          zrange=zcorr, output_filename=_p("correlation_matrix_residuals"))

    # Covariance
    utils.plot_covariance(df=cov_true, nch_per_erx=nch_per, title="Covariance (true)",
                          xtitle="channel i", ytitle="channel j", ztitle="cov(i,j)",
                          zrange=zcov, output_filename=_p("covariance_matrix_true"))
    utils.plot_covariance(df=cov_pred, nch_per_erx=nch_per, title="Covariance (prediction)",
                          xtitle="channel i", ytitle="channel j", ztitle="cov(i,j)",
                          zrange=zcov, output_filename=_p("covariance_matrix_pred"))
    utils.plot_covariance(df=cov_res, nch_per_erx=nch_per, title="Covariance (residuals)",
                          xtitle="channel i", ytitle="channel j", ztitle="cov(i,j)",
                          zrange=zcov, output_filename=_p("covariance_matrix_residuals"))
    print(f"using [pickle→plots] {bundle_path} all cov/corr graphs created.")


def add_cms_to_measurements_df(measurements_df: pd.DataFrame, cm_df: pd.DataFrame, drop_constant_cm: bool = True) -> pd.DataFrame:
    X = pd.concat([measurements_df, cm_df], axis=1)
    if drop_constant_cm and cm_df.shape[1] > 0:
        # drop CM columns with zero variance (avoid NaNs in correlation)
        cm_std = cm_df.std(axis=0)
        keep = cm_std[cm_std > 0].index
        dropped = [c for c in cm_df.columns if c not in keep]
        if dropped:
            print(f"[info] Dropping {len(dropped)} constant CM columns: {dropped}")
        X = pd.concat([measurements_df, cm_df[keep]], axis=1)
    return X

def _pivot_measurements(values: np.ndarray, channels: np.ndarray, eventid: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"value": values, "channel": channels.astype(int), "eventid": eventid.astype(int)})
    wide = df.pivot(index="eventid", columns="channel", values="value")
    wide.columns = [f"ch_{c:03d}" for c in wide.columns]
    return wide.sort_index().reindex(columns=sorted(wide.columns))
"""
def _build_input_and_cm_df(inputs_flat: np.ndarray, eventid_flat: np.ndarray, ncm: int, colnames_inputs: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if inputs_flat.shape[1] < ncm:
        raise ValueError(f"Requested ncm={ncm} but inputs have only {inputs_flat.shape[1]} columns")

    # unique event IDs with first occurrence positions, then sort by eventid
    unique_ids, first_pos = np.unique(eventid_flat, return_index=True)
    order = np.argsort(unique_ids)
    event_ids_sorted = unique_ids[order]
    rows = first_pos[order]

    # full inputs df in original column order recorded in colnames_inputs
    inputs_df = pd.DataFrame(inputs_flat[rows, :], index=event_ids_sorted, columns=colnames_inputs)

    # CM subset (preserve the original order from colnames_inputs)
    cm_cols = [c for c in colnames_inputs if c.startswith("cm_erx")]
    if len(cm_cols) != ncm:
        raise ValueError(f"Found {len(cm_cols)} CM columns by name ({cm_cols[:5]}...), but cfg.ncmchannels={ncm}.")
    cm_df = inputs_df[cm_cols]

    return (inputs_df, cm_df)
"""
def _build_input_and_cm_df(inputs_flat: np.ndarray,
                           eventid_flat: np.ndarray,
                           ncm: int,
                           colnames_inputs: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the full input and CM DataFrames from the flattened arrays.
    If a mismatch occurs between inputs_flat and colnames_inputs (e.g., 20 vs 16 features),
    automatically attempt to recover colnames.json from the next module (e.g., 0191).
    """
    if inputs_flat.shape[1] < ncm:
        raise ValueError(f"Requested ncm={ncm} but inputs have only {inputs_flat.shape[1]} columns")

    # --- Identify unique event IDs and sorting order ---
    unique_ids, first_pos = np.unique(eventid_flat, return_index=True)
    order = np.argsort(unique_ids)
    event_ids_sorted = unique_ids[order]
    rows = first_pos[order]

    # --- Try to build the DataFrame normally ---
    try:
        inputs_df = pd.DataFrame(inputs_flat[rows, :], index=event_ids_sorted, columns=colnames_inputs)

    except ValueError as e:
        print(f"[WARNING] Feature mismatch detected while building inputs DataFrame.")
        print(f"  → inputs_flat columns = {inputs_flat.shape[1]}")
        print(f"  → colnames.json length = {len(colnames_inputs)}")

        # --- Try recovery from the next module directory ---
        # Determine current working input directory if available
        import inspect, os, json
        caller_frame = inspect.stack()[1]
        local_vars = caller_frame.frame.f_locals
        input_dir = local_vars.get("input_dir", None)

        recovered = False
        if input_dir is not None and "ML_F3W_WXIH" in input_dir:
            parts = input_dir.split(os.sep)
            try:
                module_combo = parts[-4]   # e.g. ML_F3W_WXIH0190_ML_F3W_WXIH0191
                model_name = parts[-3]     # e.g. in20__512-512-512-512-64__dr0
                parent_dir = os.path.join(*parts[:-4])
            except IndexError:
                module_combo, model_name, parent_dir = None, None, None

            if module_combo and model_name:
                modules = module_combo.split("_")
                # find current module (e.g., ML_F3W_WXIH0190)
                for idx, mod in enumerate(modules):
                    if "ML_F3W_WXIH" in mod and idx + 1 < len(modules):
                        next_module = modules[idx + 1]
                        next_combo = "_".join(modules[idx:idx+2])
                        next_input_dir = os.path.join(parent_dir, next_combo, model_name, "dnn", "inputs")
                        next_json = os.path.join(next_input_dir, "colnames.json")

                        if os.path.exists(next_json):
                            with open(next_json, "r") as f:
                                next_cols = json.load(f)
                            if len(next_cols) == inputs_flat.shape[1]:
                                print(f"[RECOVER] Using colnames.json from next module: {next_module}")
                                colnames_inputs = next_cols
                                recovered = True
                                break
                            else:
                                print(f"[RECOVER] Fallback colnames.json found in {next_module} "
                                      f"but still mismatched (len={len(next_cols)}).")

        if not recovered:
            print("[RECOVER] Could not recover from next module. Truncating to smaller column count.")
            min_cols = min(inputs_flat.shape[1], len(colnames_inputs))
            inputs_flat = inputs_flat[:, :min_cols]
            colnames_inputs = colnames_inputs[:min_cols]

        # Rebuild DataFrame after recovery attempt
        inputs_df = pd.DataFrame(inputs_flat[rows, :], index=event_ids_sorted, columns=colnames_inputs)
        print(f"[RECOVER] inputs_df successfully rebuilt after mismatch.")

    # --- Extract CM columns and verify count ---
    cm_cols = [c for c in colnames_inputs if c.startswith("cm_erx")]
    if len(cm_cols) != ncm:
        raise ValueError(
            f"Found {len(cm_cols)} CM columns by name ({cm_cols[:5]}...), "
            f"but cfg.ncmchannels={ncm}."
        )

    cm_df = inputs_df[cm_cols]
    return (inputs_df, cm_df)


# ===========ANALYTIC CALCULATIONS ===================

def _analytic_bundle_path(eval_module: str, train_module: str, model_name: str) -> str:
    # create the full directory path for the analytic bundle under the performance folder
    root = os.path.join("plots", "performance", eval_module, train_module, model_name, ANALYTIC_OUTPUT_SUBDIR)
    os.makedirs(root, exist_ok=True)  # ensure that the directory exists
    return os.path.join(root, "analytic_bundle.pkl.gz")  # return the full path to the bundle file


def save_analytic_bundle(
    *,
    eval_module: str,
    train_module: str,
    model_name: str,
    nodes_per_layer: Tuple[int, ...],
    dropout_rate: float,
    cfg: ep.EvalConfig,
    split: ep.SplitData,
    variants: Dict[str, pd.DataFrame],
    variants_with_cms: Dict[str, pd.DataFrame],
    residuals: Dict[str, pd.DataFrame],
    residuals_with_cms: Dict[str, pd.DataFrame],
    k_list: Tuple[int, ...],
) -> str:
    """
    Combine analytic predictions + residuals + covariance/correlation matrices for each k into a single package.
    """
    covcorr = {}
    for key in variants_with_cms.keys():
        cov, corr = ep.compute_cov_corr(variants_with_cms[key])
        covcorr.setdefault("pred", {})[key] = {"cov": cov, "corr": corr}
    for key in residuals_with_cms.keys():
        cov, corr = ep.compute_cov_corr(residuals_with_cms[key])
        covcorr.setdefault("residual", {})[key] = {"cov": cov, "corr": corr}

    bundle = {
        "version": 1,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "meta": {
            "type": "analytic",
            "eval_module": eval_module,
            "train_module": train_module,
            "model_name": model_name,
            "nodes_per_layer": list(nodes_per_layer),
            "dropout_rate": float(dropout_rate),
            "ncmchannels": int(cfg.ncmchannels),
            "nch_per_erx": int(cfg.nch_per_erx),
            "nerx": int(cfg.nerx),
            "k_list": list(k_list),
            "source": "evaluate_performance.py",
            "fit_pool_modules": list(getattr(cfg, "modulenames_used_for_training", [])),
        },

        "indices": {
            "event_ids": split.event_ids,
            "channel_names": list(split.measurements_df.columns),
            "cm_colnames": list(split.cm_df.columns),
        },
        "frames": {
            "true": split.measurements_df,
            "cm": split.cm_df,
            "analytic": {k: variants[f"analytic_k{k}"] for k in k_list if f"analytic_k{k}" in variants},
            "residuals": {k: residuals[f"analytic_k{k}"] for k in k_list if f"analytic_k{k}" in residuals},
        },
        "covcorr": covcorr,
    }

    out_path = _analytic_bundle_path(eval_module, train_module, model_name)
    with gzip.open(out_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[analytic] wrote {out_path}")
    return out_path

def _parse_allowed(model_name: str) -> Tuple[Optional[Tuple[int, ...]], Optional[float]]:
    nodes, dr = parse_model_config(model_name)
    return nodes, dr


def _analytic_should_run_for(eval_module: str, train_module: str, model_name: str) -> bool:
    import re as _re
    filt = ANALYTIC_FILTER
    nodes, dr = _parse_allowed(model_name)

    # dropout filter
    if filt.get("dropout_allowed"):
        if dr is None or (dr not in filt["dropout_allowed"]):
            return False

    # nodes_per_layer filter
    if filt.get("nodes_per_layer"):
        if (nodes is None) or (tuple(nodes) not in [tuple(t) for t in filt["nodes_per_layer"]]):
            return False

    # eval filter
    if filt.get("eval_modules"):
        if eval_module not in filt["eval_modules"]:
            return False

    # train include/exclude regex
    inc = filt.get("train_include_regex")
    exc = filt.get("train_exclude_regex")
    if inc and not _re.search(inc, train_module):
        return False
    if exc and _re.search(exc, train_module):
        return False

    return True


def build_and_save_analytic_bundle(
    *,
    eval_module: str,
    train_module: str,
    model_name: str,
    username_load_model_from: str
) -> Optional[str]:
    """
    Use DataIO + AnalyticInferencer from evaluate_performance.py
    to compute analytic predictions and save the results as analytic_bundle.pkl.gz.
    """
    out_path = _analytic_bundle_path(eval_module, train_module, model_name)
    if (not ANALYTIC_OVERWRITE) and os.path.exists(out_path):
        print(f"[analytic] exists, skipping: {out_path}")
        return out_path

    nodes, dr = _parse_allowed(model_name)
    if nodes is None or dr is None:
        print(f"[analytic] skip (cannot parse model name): {model_name}")
        return None

    # 1) evaluate_performance configuration
    cfg = ep.EvalConfig(
        modulenames_used_for_training=_get_eval_modules_from_train_module(train_module),
        modulename_for_evaluation=eval_module,
        nodes_per_layer=list(nodes),
        dropout_rate=float(dr),
        modeltag="",
        inputfoldertag="",
        ncmchannels=12,
    )
    io = ep.DataIO(cfg=cfg)
    io.load_all()

    # --- CHANGED: split_predict = EVAL domain, split_correction = TRAIN-POOL domain ---
    split_eval = io.get_split("combined")                     # where we will PREDICT & PLOT
    split_fit  = _build_train_pool_split(                     # where we will FIT analytic params
        modnames=cfg.modulenames_used_for_training,
        base_cfg=cfg
    )


    analytic = ep.AnalyticInferencer(drop_constant_cm=True)

    # 'model_folder' is irrelevant for analytic; keep dummy
    dummy_model_folder = "."

    variants, variants_with_cms = ep.build_variants(
        split_predict=split_eval,
        split_correction=split_fit,
        cfg=cfg,
        model_folder=dummy_model_folder,
        dnn_inferencer=None,
        analytic_inferencer=analytic,
        k_list=ANALYTIC_K_LIST
    )
    residuals, residuals_with_cms = ep.make_residuals(variants, cm_df=split_eval.cm_df)


    # 3) save bundle
    return save_analytic_bundle(
        eval_module=eval_module,
        train_module=train_module,
        model_name=model_name,
        nodes_per_layer=tuple(nodes),
        dropout_rate=float(dr),
        cfg=cfg,
        split=split_eval,
        variants=variants,
        variants_with_cms=variants_with_cms,
        residuals=residuals,
        residuals_with_cms=residuals_with_cms,
        k_list=ANALYTIC_K_LIST,
    )
def run_analytic_jobs(username_load_model_from: str):
    if not ANALYTIC_ENABLE:
        print("[analytic] disabled (ANALYTIC_ENABLE=False)")
        return

    modules, models_map = discover_modules_and_models(username_load_model_from)

    # If the eval list in the filter is empty, loop through all input modules
    target_eval_modules = ANALYTIC_FILTER.get("eval_modules") or modules

    total, done = 0, 0
    for train_module, model_list in sorted(models_map.items()):
        for model_name in sorted(model_list):
            # Only run for the eval modules specified in the filter
            for eval_module in target_eval_modules:
                if not _analytic_should_run_for(eval_module, train_module, model_name):
                    continue
                total += 1
                print(f"[analytic] running for EVAL={eval_module}  TRAIN={train_module}  MODEL={model_name}")
                try:
                    path = build_and_save_analytic_bundle(
                        eval_module=eval_module,
                        train_module=train_module,
                        model_name=model_name,
                        username_load_model_from=username_load_model_from
                    )
                    if path:
                        done += 1
                except Exception as e:
                    print(f"[analytic] ERROR for ({eval_module}, {train_module}, {model_name}): {e}")

    print(f"[analytic] finished: {done}/{total} bundles created.")


#Analytic noise ratio

def _iter_all_analytic_bundles(base="plots/performance"):
    """
    Yield (eval_module, train_module, model_name, bundle_path) for all analytic bundles.
    Pattern:
      plots/performance/<EVAL>/<TRAIN>/<MODEL>/analytic/analytic_bundle.pkl.gz
    """
    pattern = os.path.join(base, "*", "*", "*", ANALYTIC_OUTPUT_SUBDIR, "analytic_bundle.pkl.gz")
    for p in glob.glob(pattern):
        parts = p.split(os.sep)
        # .../plots/performance/<EVAL>/<TRAIN>/<MODEL>/analytic/analytic_bundle.pkl.gz
        eval_module, train_module, model_name = parts[-5], parts[-4], parts[-3]
        yield eval_module, train_module, model_name, p


def _make_noise_plot_from_analytic_bundle(bundle_path: str, k: int) -> None:
    """
    Read analytic_bundle.pkl.gz and make the coherent/incoherent noise figure
    using plot_coherent_noise(), saving as:
      noise_fractions_with_ratio__analytic_k{K}.pdf
    in the same analytic/ directory as the bundle.
    """
    with gzip.open(bundle_path, "rb") as f:
        b = pickle.load(f)

    meta    = b.get("meta", {})
    frames  = b.get("frames", {})
    if not meta or not frames:
        print(f"[analytic-noise] Missing meta/frames in {bundle_path}")
        return

    true_df = frames.get("true")
    cm_df   = frames.get("cm")
    anal_map = frames.get("analytic", {}) or {}
    res_map  = frames.get("residuals", {}) or {}

    if true_df is None:
        print(f"[analytic-noise] 'true' frame is missing in {bundle_path}")
        return

    # Use analytic prediction if present; otherwise reconstruct from residuals (pred = true - residual)
    if k in anal_map:
        pred_df = anal_map[k]
    elif k in res_map:
        pred_df = true_df - res_map[k]
    else:
        print(f"[analytic-noise] Neither analytic[k={k}] nor residuals[k={k}] present in {bundle_path}")
        return

    # Keep only channel columns in numeric order (ch_000, ch_001, ...)
    ch_cols = [c for c in true_df.columns if str(c).startswith("ch_")]
    ch_cols_sorted = sorted(ch_cols, key=lambda c: int(c.split("_")[1]))
    true_df = true_df[ch_cols_sorted]
    pred_df = pred_df[ch_cols_sorted]

    # Flatten row-major to match plot_coherent_noise() expectations
    y_true = true_df.to_numpy().ravel(order="C")
    y_pred = pred_df.to_numpy().ravel(order="C")

    # Build matching eventid/chadc arrays
    ev_ids = true_df.index.to_numpy()
    nch    = len(ch_cols_sorted)
    eventid = np.repeat(ev_ids, nch)
    ch_nums = np.array([int(c.split("_")[1]) for c in ch_cols_sorted], dtype=int)
    chadc   = np.tile(ch_nums, len(ev_ids))

    # ERx geometry from meta
    nch_per  = int(meta.get("nch_per_erx", 37))
    nerx     = int(meta.get("nerx", 6))

    out_dir = os.path.dirname(bundle_path)  # .../<MODEL>/analytic
    os.makedirs(out_dir, exist_ok=True)

    # Make the plot
    plot_coherent_noise(
        y_true=y_true,
        y_pred=y_pred,
        chadc=chadc,
        eventid=eventid,
        nch_per_erx=nch_per,
        nerx=nerx,
        inputfolder="",                 # not used by the function
        plotfolder=out_dir,
        label_suffix=f"analytic_k{k}"   # this is appended before ".pdf"
    )

    # Rename to the exact requested filename pattern (drop the default suffix position)
    default_path = os.path.join(out_dir, f"noise_fractions_with_ratio_{'analytic_k'+str(k)}.pdf")
    requested_path = os.path.join(out_dir, f"noise_fractions_with_ratio__analytic_k{k}.pdf")
    if os.path.exists(default_path):
        try:
            os.replace(default_path, requested_path)
        except Exception:
            # Fallback: it might already have the correct name depending on plot_coherent_noise
            pass

    print(f"[analytic-noise] wrote {requested_path}")

def run_analytic_noise_fraction_plots(base_dir=BASE_DIR):
    """
    Iterate over analytic bundles and create noise_fractions_with_ratio__analytic_k*.pdf
    under each .../<EVAL>/<TRAIN>/<MODEL>/analytic directory.
    Respects config flags and ARCH_FILTER (model architecture).
    """
    if not ANALYTIC_NOISE_PLOTS_ENABLE:
        print("[analytic-noise] Skipped (ANALYTIC_NOISE_PLOTS_ENABLE=False).")
        return

    # Modules filter
    selected = set(ANALYTIC_NOISE_PLOTS_MODULES) if ANALYTIC_NOISE_PLOTS_MODULES else None

    # Which k values to plot
    k_list_for_plot = ANALYTIC_NOISE_PLOTS_K_LIST

    for emod, tmod, model, bundle in _iter_all_analytic_bundles(base=base_dir):

        # module filter
        if selected is not None and emod not in selected:
            continue

        # architecture filter
        if ARCH_FILTER is not None:
            arch_str = f"in20__{'-'.join(map(str, ARCH_FILTER))}"
            if arch_str not in model:
                continue

        # If user didn't force a k-list, read from the bundle meta (k_list saved when creating the analytic bundle)
        if k_list_for_plot is None:
            try:
                with gzip.open(bundle, "rb") as f:
                    bb = pickle.load(f)
                k_list = tuple(int(x) for x in bb.get("meta", {}).get("k_list", (0,)))
            except Exception:
                k_list = (0,)
        else:
            k_list = tuple(k_list_for_plot)

        print(f"[analytic-noise] eval={emod} | train={tmod} | model={model} | ks={k_list}")
        for k in k_list:
            try:
                _make_noise_plot_from_analytic_bundle(bundle, k=int(k))
            except Exception as e:
                print(f"[analytic-noise] ERROR for {bundle} (k={k}): {e}")




# ---------- Stats utilities: covariance, correlation, residuals ----------

def compute_cov(df_i: pd.DataFrame, df_j: pd.DataFrame) -> pd.DataFrame:
    """Pairwise empirical covariance with NaN-aware averaging (events x channels)."""
    print(f"--> Computing a covariance matrix from {df_i.shape} and {df_j.shape}")
    # mask of valid entries (NaN = missing)
    mask_i = df_i.notna().astype(float)
    mask_j = df_j.notna().astype(float)
    X_i = df_i.fillna(0.0).to_numpy()
    X_j = df_j.fillna(0.0).to_numpy()
    M_i = mask_i.to_numpy()
    M_j = mask_j.to_numpy()

    # per-pair counts and sums
    N = M_i.T @ M_j                    # valid-event counts per channel pair
    S = X_i.T @ X_j                    # sum of products (zeros where NaN)

    # average only over valid events
    with np.errstate(invalid="ignore", divide="ignore"):
        C = S / N
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    return pd.DataFrame(C, index=df_i.columns, columns=df_j.columns)


def corr_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert covariance to correlation (safe when diagonal has zeros)."""
    d = np.diag(cov.to_numpy())
    inv = np.zeros_like(d, dtype=float)
    pos = d > 0
    inv[pos] = 1.0 / np.sqrt(d[pos])
    D = np.diag(inv)
    R = D @ cov.to_numpy() @ D
    # numerical guard
    np.clip(R, -1.0, 1.0, out=R)
    return pd.DataFrame(R, index=cov.index, columns=cov.columns)



def compute_cov_corr(df: pd.DataFrame, log_file: str = "zero_std_channels.txt") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper returning (cov, corr).
    Additionally logs channels with std=0 (only ch_* channels).
    """
    cov = compute_cov(df_i=df, df_j=df)
    corr = corr_from_cov(cov)

    # Extract variance from diagonal
    diag = np.diag(cov.to_numpy())
    stds = np.sqrt(diag)

    # Check only measurement channels (prefix "ch_")
    zero_std_channels = [
        cov.index[i] for i, s in enumerate(stds)
        if s == 0 and cov.index[i].startswith("ch_")
    ]

    if zero_std_channels:
        with open(log_file, "a") as f:   # append mode
            f.write("\n=== New run (ch_* only) ===\n")
            for ch in zero_std_channels:
                f.write(ch + "\n")
        print(f"[info] Found {len(zero_std_channels)} ch_* channels with std=0. See '{log_file}'.")

    return (cov, corr)


def plot_covariance(df, nch_per_erx, title, xtitle, ytitle, ztitle, output_filename, zrange=(-1., 1.)):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot 2D heatmap
    im = ax.pcolormesh(
        df.columns,  # x bin edges
        df.index,    # y bin edges
        df.values,   # 2D values
        shading='auto',
        cmap='coolwarm',   # or 'RdBu', 'coolwarm', 'plasma', etc.,
        vmin=zrange[0],
        vmax=zrange[1]
    )

    # Draw dashed lines every `nch_per_erx` channels
    n_channels = df.shape[0]
    for i in range(nch_per_erx, n_channels, nch_per_erx):
        ax.axhline(i-0.5, color='black', linestyle='--', linewidth=0.7)
        ax.axvline(i-0.5, color='black', linestyle='--', linewidth=0.7)

    ticks = np.arange(0, n_channels, nch_per_erx)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)

    # Labels and styling
    ax.set_title(title)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(ztitle)

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved 2-d plot {output_filename}")
    plt.close()

# Aggregation stores
layer_results = defaultdict(list)
detailed_results = defaultdict(list)
dropout_group_results = defaultdict(list)
dropout_coherent_group_results = defaultdict(list)
coherent_results = defaultdict(list)
coherent_detailed_results = defaultdict(list)
per_module_fractional_means = defaultdict(list)
per_module_coherent_means = defaultdict(list)

# ----- Parsing / grouping -----
def parse_model_config(model_name):
    try:
        parts = model_name.split("__")
        nodes_str = parts[1]
        nodes = tuple(map(int, nodes_str.split("-")))
        dr = float(parts[2].replace("dr", ""))
        return nodes, dr
    except Exception:
        return None, None
_MODULE_RE = re.compile(r"(ML_F3W_WXIH\d+)")

def is_self(test_module, train_module):
    return test_module in _MODULE_RE.findall(train_module)

# ----- Results I/O -----
def save_result_to_txt(test_module, train_module, nodes_per_layer, dropout_rate, frac_impr_mean, coh_ratio_mean):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    entry = f"{test_module},{train_module},{'-'.join(map(str, nodes_per_layer))},{dropout_rate},{frac_impr_mean},{coh_ratio_mean}\n"
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            if entry.strip() in (line.strip() for line in f):
                print(f"Duplicate prevented: {test_module}/{train_module}/{nodes_per_layer}/dr{dropout_rate}")
                return
    with open(RESULTS_FILE, "a") as f:
        f.write(entry)

def register_result(test_module, train_module, nodes_per_layer, dropout_rate, frac_impr_mean, coh_ratio_mean):
    """Record metrics into file; add to aggregates only if ARCH passes."""
    save_result_to_txt(test_module, train_module, nodes_per_layer, dropout_rate, frac_impr_mean, coh_ratio_mean)

    layer_key = tuple(nodes_per_layer)
    if not _arch_ok(layer_key):
        return  # keep results.txt full, but skip aggregates for charts

    per_module_fractional_means[test_module].append(frac_impr_mean)
    per_module_coherent_means[test_module].append(coh_ratio_mean)

    layer_results[(test_module, train_module, layer_key)].append(frac_impr_mean)
    coherent_results[(test_module, train_module, layer_key)].append(coh_ratio_mean)

    detailed_results[(test_module, train_module, layer_key)].append((dropout_rate, frac_impr_mean))
    coherent_detailed_results[(test_module, train_module, layer_key)].append((dropout_rate, coh_ratio_mean))

    group_type = "SELF" if is_self(test_module, train_module) else "CROSS"
    dropout_group_results[(group_type, dropout_rate)].append(frac_impr_mean)
    dropout_coherent_group_results[(group_type, dropout_rate)].append(coh_ratio_mean)

def load_existing_results():
    """Warm start aggregation from RESULTS_FILE (apply ARCH filter to aggregates)."""
    if not os.path.exists(RESULTS_FILE):
        return
    with open(RESULTS_FILE, "r") as f:
        for line in f:
            try:
                test_module, train_module, layer_str, dropout, frac_impr, coh_ratio = line.strip().split(",")
                layer_tuple = tuple(map(int, layer_str.split("-")))
                dropout_val = float(dropout)
                frac_impr_mean = float(frac_impr)
                coh_ratio_mean = float(coh_ratio)

                # always fill skip keys so we don't re-run evaluated combos
                if ENABLE_RESULTS_TXT_SKIP:
                    existing_result_keys.add((test_module, train_module, layer_tuple, dropout_val))

                # apply ARCH filter to in-memory aggregates only
                if not _arch_ok(layer_tuple):
                    continue

                group_type = "SELF" if is_self(test_module, train_module) else "CROSS"
                dropout_group_results[(group_type, dropout_val)].append(frac_impr_mean)
                dropout_coherent_group_results[(group_type, dropout_val)].append(coh_ratio_mean)

                layer_results[(test_module, train_module, layer_tuple)].append(frac_impr_mean)
                coherent_results[(test_module, train_module, layer_tuple)].append(coh_ratio_mean)

                per_module_fractional_means[test_module].append(frac_impr_mean)
                per_module_coherent_means[test_module].append(coh_ratio_mean)

            except ValueError:
                continue

# ----- Discovery / filtering -----
def discover_modules_and_models(username_load_model_from):
    """List input modules and model folders."""
    input_base = f"/eos/user/{os.getenv('USER')[0]}/{os.getenv('USER')}/hgcal/dnn_inputs"
    modules = [d for d in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, d))]
    models_base = f"/eos/user/{username_load_model_from[0]}/{username_load_model_from}/hgcal/dnn_models"
    models = {}
    for module in os.listdir(models_base):
        module_path = os.path.join(models_base, module)
        if not os.path.isdir(module_path):
            continue
        models[module] = [d for d in os.listdir(module_path) if os.path.isdir(os.path.join(module_path, d))]
    return modules, models


"""
def filter_existing_plots(modules, models, output_base="plots/performance"):
    #Skip combos already logged (and/or with existing folders).
    filtered = {}
    for test_module in modules:
        combos = []
        for train_module, model_list in models.items():
            for model_name in model_list:
                nodes_per_layer, dropout_rate = parse_model_config(model_name)
                if nodes_per_layer is None or dropout_rate is None:
                    continue
                key = (test_module, train_module, tuple(nodes_per_layer), dropout_rate)
                plot_path = os.path.join(output_base, test_module, train_module, model_name)
                has_results_entry = key in existing_result_keys if ENABLE_RESULTS_TXT_SKIP else False
                has_folder = os.path.exists(plot_path) if ENABLE_FOLDER_SKIP else False
                if (ENABLE_RESULTS_TXT_SKIP and has_results_entry) or (ENABLE_FOLDER_SKIP and has_folder):
                    continue
                combos.append((train_module, model_name))
        if combos:
            filtered[test_module] = combos
    return filtered
"""
def filter_existing_plots(modules, models, output_base="plots/performance"):
    """
    Skip combinations that are already logged (and/or have existing folders).
    Added feature: if a module (e.g., ML_F3W_WXIH0190) contains a corrupted or
    incomplete colnames.json file (e.g., 16 features instead of 20),
    automatically fall back to the next module (e.g., ML_F3W_WXIH0191)
    for the same model_name and train_module.
    """
    filtered = {}

    for i, test_module in enumerate(modules):
        combos = []
        for train_module, model_list in models.items():
            for model_name in model_list:
                # Parse model configuration (layer structure + dropout)
                nodes_per_layer, dropout_rate = parse_model_config(model_name)
                if nodes_per_layer is None or dropout_rate is None:
                    continue

                key = (test_module, train_module, tuple(nodes_per_layer), dropout_rate)
                plot_path = os.path.join(output_base, test_module, train_module, model_name)

                has_results_entry = key in existing_result_keys if ENABLE_RESULTS_TXT_SKIP else False
                has_folder = os.path.exists(plot_path) if ENABLE_FOLDER_SKIP else False

                # Skip combinations that already have results or output folders
                if (ENABLE_RESULTS_TXT_SKIP and has_results_entry) or (ENABLE_FOLDER_SKIP and has_folder):
                    continue

                # --- Sanity check: verify colnames.json validity ---
                input_dir = os.path.join(plot_path, "dnn", "inputs")
                colnames_path = os.path.join(input_dir, "colnames.json")

                # If colnames.json exists, check its feature count
                if os.path.exists(colnames_path):
                    try:
                        with open(colnames_path, "r") as f:
                            cols = json.load(f)
                        # Known corrupted case: 16 features instead of 20
                        if len(cols) < 20:
                            raise ValueError("insufficient feature count")
                    except Exception as e:
                        print(f"[filter] Detected invalid colnames in {test_module}: {e}")

                        # Try to use the next module in sequence as fallback
                        if i + 1 < len(modules):
                            next_module = modules[i + 1]
                            alt_input_dir = os.path.join(output_base, next_module, train_module, model_name, "dnn", "inputs")
                            alt_colnames = os.path.join(alt_input_dir, "colnames.json")

                            if os.path.exists(alt_colnames):
                                with open(alt_colnames, "r") as f:
                                    alt_cols = json.load(f)
                                # Use fallback only if feature count matches the expected one
                                if len(alt_cols) == 20:
                                    print(f"[filter] Using colnames.json from {next_module} "
                                          f"as fallback for {test_module}")
                                    combos.append((train_module, model_name))
                                    continue
                                else:
                                    print(f"[filter] Fallback colnames.json in {next_module} "
                                          f"is also invalid (len={len(alt_cols)}). Skipping.")
                                    continue
                            else:
                                print(f"[filter] No valid fallback found for {test_module}. Skipping.")
                                continue
                        else:
                            print(f"[filter] {test_module} is the last module. Cannot fallback.")
                            continue

                # If everything is fine, keep this combo
                combos.append((train_module, model_name))

        if combos:
            filtered[test_module] = combos

    return filtered

# ----- Metrics (from plot_performance) -----
def _infer_erx_params(modulename_for_evaluation: str):
    """Return (channels_per_erx, num_erx)."""
    return (37, 6) if modulename_for_evaluation.startswith("ML") else (74, 12)

def _safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.mean(x)) if x.size else float("nan")

def _bundle_pickle_path(plotfolder: str) -> str:
    """Unified location: same folder for each (eval/train/model)."""
    os.makedirs(plotfolder, exist_ok=True)
    return os.path.join(plotfolder, "predictions_bundle.pkl.gz")

def save_predictions_bundle(
    plotfolder: str,
    *,
    # metadata
    eval_module: str,
    train_module: str,
    model_name: str,
    nodes_per_layer: list,
    dropout_rate: float,
    ncmchannels: int,
    nch_per_erx: int,
    nerx: int,
    # indices & names
    eventid_combined: np.ndarray,
    chadc_combined: np.ndarray,
    colnames_inputs: list,
    # core matrices (event×channel)
    meas_true_df: pd.DataFrame,
    meas_pred_df: pd.DataFrame,
    cm_df: pd.DataFrame,
    # derivatives
    residual_df: pd.DataFrame,
    cov_true: pd.DataFrame,
    corr_true: pd.DataFrame,
    cov_pred: pd.DataFrame,
    corr_pred: pd.DataFrame,
    cov_res: pd.DataFrame,
    corr_res: pd.DataFrame,
) -> str:
    """
    Store all necessary data into a single compressed pickle file.
    This is sufficient to reproduce plots and metrics without re-running inference.
    """
    bundle = {
        "version": 1,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "meta": {
            "eval_module": eval_module,
            "train_module": train_module,
            "model_name": model_name,
            "nodes_per_layer": list(nodes_per_layer),
            "dropout_rate": float(dropout_rate),
            "ncmchannels": int(ncmchannels),
            "nch_per_erx": int(nch_per_erx),
            "nerx": int(nerx),
        },
        "indices": {
            "eventid": np.asarray(eventid_combined, dtype=np.int64),
            "chadc":   np.asarray(chadc_combined,  dtype=np.int32),
            "channel_names": list(meas_true_df.columns),
            "cm_colnames": list(cm_df.columns),
            "input_colnames": list(colnames_inputs),
        },
        "frames": {
            # event×channel tables
            "true": meas_true_df,
            "pred": meas_pred_df,
            "residual": residual_df,
            "cm": cm_df,
        },
        "covcorr": {
            "true": {"cov": cov_true, "corr": corr_true},
            "pred": {"cov": cov_pred, "corr": corr_pred},
            "residual": {"cov": cov_res, "corr": corr_res},
        },
    }

    out_path = _bundle_pickle_path(plotfolder)
    with gzip.open(out_path, "wb") as f:
        # pandas DataFrames are naturally pickled
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[pickle] Saved predictions bundle → {out_path}")
    return out_path


def evaluate_model_and_compute_metrics(modulename_for_evaluation: str, train_module: str, model_name: str, username_load_model_from: str, *, write_bundle: bool = True, write_plots: bool = False):
    """Load model, run inference, compute (frac_impr_mean, coh_ratio_mean)."""
    device = torch.device("cpu")
    nodes_per_layer, dropout_rate = parse_model_config(model_name)
    assert nodes_per_layer and dropout_rate is not None, f"Invalid model name: {model_name}"

    nch_per_erx, nerx = _infer_erx_params(modulename_for_evaluation)
    """
    m_in = re.match(r"in(\d+)", model_name)
    if not m_in:
        raise ValueError(f"Cannot infer ncmchannels from model name: {model_name}")
    ncmchannels = int(m_in.group(1))
    """

    inputfolder = f"/eos/user/{username_load_model_from[0]}/{username_load_model_from}/hgcal/dnn_inputs/{modulename_for_evaluation}"
    modelfolder = f"/eos/user/{username_load_model_from[0]}/{username_load_model_from}/hgcal/dnn_models/{train_module}/{model_name}"
    plotfolder  = f"plots/performance/{modulename_for_evaluation}/{train_module}/{model_name}"
    os.makedirs(plotfolder, exist_ok=True)

    # --- BUNDLE CHECK ADDED ---
    bundle_path = os.path.join(plotfolder, "predictions_bundle.pkl.gz")
    if os.path.exists(bundle_path):
        import gzip, pickle
        print(f"[evaluate] Bundle found: {bundle_path}, skipping recomputation.")
        with gzip.open(bundle_path, "rb") as f:
            bundle = pickle.load(f)

        # Load covariance/correlation matrices from bundle
        cov_true = bundle["covcorr"]["true"]["cov"]
        corr_true = bundle["covcorr"]["true"]["corr"]
        cov_pred = bundle["covcorr"]["pred"]["cov"]
        corr_pred = bundle["covcorr"]["pred"]["corr"]
        cov_res  = bundle["covcorr"]["residual"]["cov"]
        corr_res = bundle["covcorr"]["residual"]["corr"]

        meas_true = bundle["frames"]["true"]
        meas_pred = bundle["frames"]["pred"]
        residual_meas = bundle["frames"]["residual"]

        print("[evaluate] Loaded cov/corr matrices from bundle.")
        """
        # --- Debug prints (commented out) ---
        print(f"[DEBUG BUNDLE] TRUE cov shape={cov_true.shape}, corr shape={corr_true.shape}")
        print(cov_true.iloc[0:3, 0:3].round(4))
        print(corr_true.iloc[0:3, 0:3].round(4))
        print(f"[DEBUG BUNDLE] DNN cov shape={cov_pred.shape}, corr shape={corr_pred.shape}")
        print(cov_pred.iloc[0:3, 0:3].round(4))
        print(corr_pred.iloc[0:3, 0:3].round(4))
        print(f"[DEBUG BUNDLE] RESIDUAL cov shape={cov_res.shape}, corr shape={corr_res.shape}")
        print(cov_res.iloc[0:3, 0:3].round(4))
        print(corr_res.iloc[0:3, 0:3].round(4))
        """
        # Compute metrics also when reading from bundle
        true_np = np.asarray(meas_true, dtype=float)
        res_np  = np.asarray(residual_meas, dtype=float)

        rms_true = np.sqrt(np.mean(true_np**2, axis=0))
        rms_res  = np.sqrt(np.mean(res_np**2,  axis=0))
        denom = np.where(rms_true > 0, rms_true, np.nan)
        frac_impr = 1.0 - (rms_res / denom)
        frac_impr_mean = float(np.nanmean(frac_impr))

        nch_per_erx = int(bundle["meta"]["nch_per_erx"])
        nerx        = int(bundle["meta"]["nerx"])

        def _dir_alt_sums(B):
            direct = B.sum(axis=1)
            alternating = B[:, ::2].sum(axis=1) - B[:, 1::2].sum(axis=1)
            return direct, alternating

        def _trms(x):
            x = np.asarray(x, float)
            return float(np.sqrt(np.mean(x**2))) if x.size else float("nan")

        def _coh_inc(B, n):
            d, a = _dir_alt_sums(B)
            rms_d, rms_a = _trms(d), _trms(a)
            delta = rms_d**2 - rms_a**2
            inc = rms_a / np.sqrt(n)
            coh = np.sign(delta) * np.sqrt(abs(delta)) / n
            return coh, inc

        inc_ratios = []
        ncols = true_np.shape[1]
        full_n = min(ncols, nch_per_erx * nerx)
        for erx in range(nerx):
            i0 = erx * nch_per_erx
            i1 = min((erx + 1) * nch_per_erx, full_n)
            if i0 >= i1:
                continue
            T = true_np[:, i0:i1]
            R = res_np[:,  i0:i1]
            _, inc_t = _coh_inc(T, i1 - i0)
            _, inc_c = _coh_inc(R, i1 - i0)
            if inc_t > 0:
                inc_ratios.append(inc_c / inc_t)

        coh_ratio_mean = float(np.mean(inc_ratios)) if inc_ratios else float("nan")

        return (frac_impr_mean, coh_ratio_mean)

    # ======= SIMPLE GUARD: empty/missing input folder → skip early =======
    # If the input folder does not exist or is empty, skip this module entirely.
    if (not os.path.isdir(inputfolder)) or (len(os.listdir(inputfolder)) == 0):
        print(f"[skip] Empty or missing input folder: {inputfolder} → skipping {modulename_for_evaluation}")
        return None

    # Try to open colnames.json; if missing, skip cleanly.
    try:
        with open(f"{inputfolder}/colnames.json") as f:
            colnames_inputs = json.load(f)
    except FileNotFoundError:
        print(f"[skip] Missing colnames.json for {modulename_for_evaluation} → skipping")
        return None
    # ======= /guard =======


    cm_cols = [c for c in colnames_inputs if c.startswith("cm_erx")]
    ncmchannels = len(cm_cols)


    # --- Load inputs and model ---
    inputs_train  = np.load(f"{inputfolder}/inputs_train.npy")
    inputs_val    = np.load(f"{inputfolder}/inputs_val.npy")
    targets_train = np.load(f"{inputfolder}/targets_train.npy")
    targets_val   = np.load(f"{inputfolder}/targets_val.npy")

    # --- Convert to tensors ---
    X_train = torch.tensor(inputs_train,  dtype=torch.float32).to(device)
    X_val   = torch.tensor(inputs_val,    dtype=torch.float32).to(device)
    y_train = torch.tensor(targets_train, dtype=torch.float32).squeeze().to(device)
    y_val   = torch.tensor(targets_val,   dtype=torch.float32).squeeze().to(device)

    # --- Load full chadc & split it by saved indices ---
    chadc_full   = np.load(f"{inputfolder}/chadc.npy").astype(int).squeeze()
    eventid_full = np.load(f"{inputfolder}/eventid.npy").astype(int).squeeze()
    train_idx    = np.load(f"{inputfolder}/indices_train.npy")
    val_idx      = np.load(f"{inputfolder}/indices_val.npy")
    chadc_train  = chadc_full[train_idx]
    chadc_val    = chadc_full[val_idx]
    eventid_train = eventid_full[train_idx]
    eventid_val   = eventid_full[val_idx]

    # --- Instantiate & load model ---
    model = models.DNNFlex(input_dim=X_train.shape[1], nodes_per_layer=nodes_per_layer, dropout_rate=dropout_rate, tag="")
    pth = f"{modelfolder}/regression_dnn_best.pth"
    if not os.path.exists(pth):
        raise FileNotFoundError(f"Model weights not found: {pth}")
    model.load_state_dict(torch.load(pth, map_location=device))
    model.to(device).eval()

    # --- Predict separately for train and val ---
    with torch.no_grad():
        y_pred_train = model(X_train).squeeze().cpu().numpy()
        y_pred_val   = model(X_val).squeeze().cpu().numpy()

    # bring true y back to numpy
    y_train_np = y_train.detach().cpu().numpy().squeeze()
    y_val_np   = y_val.detach().cpu().numpy().squeeze()

    # --- Build the “combined” arrays for diagnostics ---
    inputs_combined  = np.concatenate([inputs_train, inputs_val], axis=0)
    y_true_combined  = np.concatenate([y_train_np,  y_val_np],    axis=0)
    y_pred_combined  = np.concatenate([y_pred_train, y_pred_val], axis=0)
    chadc_combined   = np.concatenate([chadc_train,  chadc_val],  axis=0)
    eventid_combined = np.concatenate([eventid_train, eventid_val], axis=0)

    # Event x Channel DataFrames
    meas_true = _pivot_measurements(values=y_true_combined, channels=chadc_combined, eventid=eventid_combined)
    meas_pred = _pivot_measurements(values=y_pred_combined, channels=chadc_combined, eventid=eventid_combined)

    _, cm_df = _build_input_and_cm_df(inputs_flat=inputs_combined,
                                      eventid_flat=eventid_combined,
                                      ncm=ncmchannels,
                                      colnames_inputs=colnames_inputs)

    df_to_compute_cov_true = add_cms_to_measurements_df(measurements_df=meas_true, cm_df=cm_df, drop_constant_cm=False)
    df_to_compute_cov_pred = add_cms_to_measurements_df(measurements_df=meas_pred, cm_df=cm_df, drop_constant_cm=False)

    #residuals
    residual_meas = meas_true - meas_pred

    df_to_compute_cov_residuals = add_cms_to_measurements_df(
        measurements_df=residual_meas,
        cm_df=cm_df,
        drop_constant_cm=False
    )

    # === Corr & Cov (TRUE / RESIDUALS) ===
    cov_true, corr_true = compute_cov_corr(df_to_compute_cov_true)
    cov_pred, corr_pred = compute_cov_corr(df_to_compute_cov_pred)
    cov_res,  corr_res  = compute_cov_corr(df_to_compute_cov_residuals)
    print(f"[DEBUG KOD2] TRUE cov shape={cov_true.shape}, corr shape={corr_true.shape}")
    print(cov_true.iloc[:3, :3].round(4))
    print(corr_true.iloc[:3, :3].round(4))

    print(f"[DEBUG KOD2] DNN cov shape={cov_pred.shape}, corr shape={corr_pred.shape}")
    print(cov_pred.iloc[:3, :3].round(4))
    print(corr_pred.iloc[:3, :3].round(4))

    print(f"[DEBUG KOD2] RESIDUAL cov shape={cov_res.shape}, corr shape={corr_res.shape}")
    print(cov_res.iloc[:3, :3].round(4))
    print(corr_res.iloc[:3, :3].round(4))

    # --- Pickle bundle: save all computed data into a single file ---
    # --- Save bundle only if allowed ---
    bundle_path = os.path.join(plotfolder, "predictions_bundle.pkl.gz")
    if write_bundle:
        save_predictions_bundle(
            plotfolder=plotfolder,
            eval_module=modulename_for_evaluation,
            train_module=train_module,
            model_name=model_name,
            nodes_per_layer=nodes_per_layer,
            dropout_rate=dropout_rate,
            ncmchannels=ncmchannels,
            nch_per_erx=nch_per_erx,
            nerx=nerx,
            eventid_combined=eventid_combined,
            chadc_combined=chadc_combined,
            colnames_inputs=colnames_inputs,
            meas_true_df=meas_true,
            meas_pred_df=meas_pred,
            cm_df=cm_df,
            residual_df=residual_meas,
            cov_true=cov_true,  corr_true=corr_true,
            cov_pred=cov_pred,  corr_pred=corr_pred,
            cov_res=cov_res,    corr_res=corr_res,
        )

    """
    cov_true  = cov_true.sort_index().sort_index(axis=1)
    corr_true = corr_true.sort_index().sort_index(axis=1)
    cov_pred  = cov_pred.sort_index().sort_index(axis=1)
    corr_pred = corr_pred.sort_index().sort_index(axis=1) 
    cov_res   = cov_res.sort_index().sort_index(axis=1)
    corr_res  = corr_res.sort_index().sort_index(axis=1)
    """
    # Output results
    out_true_corr = f"{plotfolder}/correlation_matrix_true.pdf"
    out_res_corr  = f"{plotfolder}/correlation_matrix_residuals.pdf"
    out_pred_corr = f"{plotfolder}/correlation_matrix_pred.pdf"
    out_true_cov  = f"{plotfolder}/covariance_matrix_true.pdf"
    out_pred_cov  = f"{plotfolder}/covariance_matrix_pred.pdf"
    out_res_cov   = f"{plotfolder}/covariance_matrix_residuals.pdf"

    if write_plots:
        # Correlation plots
        utils.plot_covariance(
            df=corr_true, nch_per_erx=nch_per_erx, title="Correlation (true)",
            xtitle="channel i", ytitle="channel j", ztitle="corr(i,j)",
            zrange=(-1., 1.), output_filename=out_true_corr
        )
        utils.plot_covariance(
            df=corr_pred, nch_per_erx=nch_per_erx, title="Correlation (prediction)",
            xtitle="channel i", ytitle="channel j", ztitle="corr(i,j)",
            zrange=(-1., 1.), output_filename=out_pred_corr
        )
        utils.plot_covariance(
            df=corr_res, nch_per_erx=nch_per_erx, title="Correlation (residuals)",
            xtitle="channel i", ytitle="channel j", ztitle="corr(i,j)",
            zrange=(-1., 1.), output_filename=out_res_corr
        )

        # Covariance plots
        utils.plot_covariance(
            df=cov_true, nch_per_erx=nch_per_erx, title="Covariance (true)",
            xtitle="channel i", ytitle="channel j", ztitle="cov(i,j)",
            zrange=(-4., 4.), output_filename=out_true_cov
        )
        utils.plot_covariance(
            df=cov_pred, nch_per_erx=nch_per_erx, title="Covariance (prediction)",
            xtitle="channel i", ytitle="channel j", ztitle="cov(i,j)",
            zrange=(-4., 4.), output_filename=out_pred_cov
        )
        utils.plot_covariance(
            df=cov_res, nch_per_erx=nch_per_erx, title="Covariance (residuals)",
            xtitle="channel i", ytitle="channel j", ztitle="cov(i,j)",
            zrange=(-4., 4.), output_filename=out_res_cov
        )


    # Infer number of ADC channels from chadc
    n_channels = int(chadc_combined.max()) + 1
    n_samples  = len(y_true_combined)
    assert n_samples % n_channels == 0, "Mismatch between samples and channels"

    n_events = n_samples // n_channels

    # Reshape predictions and truths
    y_combined_2d      = y_true_combined.reshape((n_events, n_channels))
    y_pred_combined_2d = y_pred_combined.reshape((n_events, n_channels))
    residual_2d        = y_combined_2d - y_pred_combined_2d

    # Compute per-channel RMS
    rms_true_per_channel = np.sqrt(np.mean(y_combined_2d**2, axis=0))
    rms_corrected_per_channel = np.sqrt(np.mean(residual_2d**2, axis=0))
    std_true_per_channel = np.std(y_combined_2d, axis=0)
    std_corrected_per_channel = np.std(residual_2d, axis=0)

    # fractional improvement  (1 − corrected / uncorrected)
    frac_impr = 1.0 - (rms_corrected_per_channel / rms_true_per_channel)
    frac_impr_mean  = float(frac_impr.mean())

    # ---- Coherent noise ratio (corr/true) – mirror of plot_coherent_noise ----
    coh_true_list, coh_corr_list = [], []
    for erx in range(nerx):

        # order rows by (eventid, chadc) – minor key first!
        order = np.lexsort((chadc_combined, eventid_combined))     # minor key first!
        chadc_ordered   = chadc_combined[order]
        eventid_ordered = eventid_combined[order]
        y_true_ordered  = y_true_combined[order]
        y_pred_ordered  = y_pred_combined[order]

        # ------- pick rows that belong to this ERx -----------------------
        mask  = (chadc_ordered >=  erx      * nch_per_erx) & (chadc_ordered <  (erx + 1) * nch_per_erx)
        idxs  = np.where(mask)[0]                 # integer positions
        if idxs.size == 0:
            continue

        adc_true = y_true_ordered[idxs]
        adc_pred = y_pred_ordered[idxs]
        adc_corr = adc_true - adc_pred

        # ── reshape to (n_events, 37) ────────────────────────────────────
        n_rows = adc_true.size
        assert n_rows % nch_per_erx == 0, "rows not multiple of nch_per_erx"
        n_evt  = n_rows // nch_per_erx

        adc_true_2d = adc_true.reshape(n_evt, nch_per_erx)
        adc_corr_2d = adc_corr.reshape(n_evt, nch_per_erx)

        # ── per-event direct & alternating sums ─────────────────────────
        dir_sums_true = adc_true_2d.sum(axis=1)                       # shape (n_evt,)
        alt_sums_true = adc_true_2d[:, ::2].sum(axis=1) - adc_true_2d[:, 1::2].sum(axis=1)

        dir_sums_corr = adc_corr_2d.sum(axis=1)
        alt_sums_corr = adc_corr_2d[:, ::2].sum(axis=1) - adc_corr_2d[:, 1::2].sum(axis=1)

        # variance difference
        delta_true = dir_sums_true.var() - alt_sums_true.var()
        delta_corr = dir_sums_corr.var() - alt_sums_corr.var()

        # incoherent and coherent components
        inc_noise_true = np.std(alt_sums_true) / math.sqrt(nch_per_erx)
        coh_noise_true = np.sign(delta_true) * math.sqrt(abs(delta_true)) / nch_per_erx

        inc_noise_corr = np.std(alt_sums_corr) / math.sqrt(nch_per_erx)
        coh_noise_corr = np.sign(delta_corr) * math.sqrt(abs(delta_corr)) / nch_per_erx

        # store only coherent parts for the ratio
        coh_true_list.append(coh_noise_true)
        coh_corr_list.append(coh_noise_corr)

    # --- ratios corr / true -------------------------------------------------
    coh_true_arr = np.array(coh_true_list)
    coh_corr_arr = np.array(coh_corr_list)
    coh_ratio    = coh_corr_arr / coh_true_arr
    coh_ratio_mean = float(np.mean(coh_ratio))

    # --- Log variance and std values for TRUE / PRED / RESIDUAL ---
    log_file = "channel_stats.txt"
    with open(log_file, "a") as f:
        f.write("\n\n=====================================\n")
        f.write(f"Module: {modulename_for_evaluation} | Train: {train_module} | Model: {model_name}\n")
        f.write("=====================================\n")

        for label, cov in [
            ("TRUE (measured)", cov_true),
            ("PRED (DNN prediction)", cov_pred),
            ("RESIDUAL (true - pred)", cov_res),
        ]:
            f.write(f"\n--- {label} ---\n")
            diag = np.diag(cov.to_numpy())
            stds = np.sqrt(diag)
            for col, var, std in zip(cov.index, diag, stds):
                f.write(f"{col:10s}  variance={var:.6e}  std={std:.6e}\n")

    print(f"[log] Variance/std values (TRUE, PRED, RESIDUAL) saved to {log_file}")

    return frac_impr_mean, coh_ratio_mean

def _plot_basic_diagnostics(plotfolder, y_true_2d, resid_2d, rms_true, rms_corr, frac_impr):
    """Lightweight sanity plots."""
    os.makedirs(plotfolder, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(rms_true, bins=30, alpha=0.6, label="Uncorrected", color='gray')
    plt.hist(rms_corr, bins=30, alpha=0.6, label="Corrected",  color='tomato')
    plt.xlabel("RMS (ADC)")
    plt.ylabel("Channels")
    plt.title("Per-channel RMS")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plotfolder}/rms_comparison_per_channel.pdf")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(frac_impr, bins=20)
    plt.xlabel(r"1 - RMS$_{corr}$/RMS$_{uncorr}$")
    plt.ylabel("Channels")
    plt.title("Fractional RMS improvement")
    plt.axvline(np.nanmean(frac_impr), color="k", ls="--", label=f"mean={np.nanmean(frac_impr):.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plotfolder}/rms_frac_improvement_per_channel.pdf")
    plt.close()

# ===== read results.txt for graphs =====
def _iter_results_txt_rows():
    """Yield rows from results.txt."""
    if not os.path.exists(RESULTS_FILE):
        return
    with open(RESULTS_FILE, "r") as f:
        for line in f:
            try:
                test_module, train_module, layer_str, dropout, frac_impr, coh_ratio = line.strip().split(",")
                layer_tuple = tuple(map(int, layer_str.split("-")))
                yield (test_module, train_module, layer_tuple, float(dropout), float(frac_impr), float(coh_ratio))
            except Exception:
                continue

def _count_trained_modules(train_module_name):
    """Count ML_* tokens in train_module."""
    return len(re.findall(r'ML_[A-Z0-9]+', train_module_name))

# ===== Coherent noise Ratio vs Dropout =====
def plot_new_dropout_vs_coherent_by_train():
    """Per train_module: dropout vs coherent; curves for eval=0190 and eval=0198.
       Output now saved under plots/performance/ with train model in filename."""
    data = defaultdict(lambda: defaultdict(list))  # {train: {eval: [(dr, coh), ...]}}
    for test_mod, train_mod, layer_t, dr, _frac, coh in _iter_results_txt_rows():
        if layer_t != ARCH_FILTER:
            continue
        if test_mod not in ("ML_F3W_WXIH0190", "ML_F3W_WXIH0198"):
            continue
        data[train_mod][test_mod].append((dr, coh))

    base_dir = os.path.join("plots", "performance")
    os.makedirs(base_dir, exist_ok=True)

    for train_mod, eval_map in data.items():
        if not eval_map:
            continue
        safe_train = _sanitize(train_mod)
        out_path = os.path.join(
            base_dir,
            f"coh_vs_dropout__train_{safe_train}.pdf"
        )

        plt.figure(figsize=(8, 6), dpi=300)
        for eval_tag, series in sorted(eval_map.items()):
            if not series:
                continue
            series = sorted(series, key=lambda x: x[0])
            x = [d for d, _ in series]
            y = [c for _, c in series]
            label = "ML_F3W_WXIH0190" if eval_tag.endswith("0190") else "ML_F3W_WXIH0198"
            plt.plot(x, y, "-D", linewidth=2.0, markersize=7, label=label)
            for xi, yi in zip(x, y):
                plt.text(xi, yi + 0.02, f"{yi:.2f}", ha="center", fontsize=9)

        plt.xlabel("Dropout rate")
        plt.ylabel("Coherent noise ratio (corr/true)")
        plt.title("Dropout vs coherent noise ratio")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

def plot_new_numtrained_vs_coherent_by_dropout():
    """Per dropout: #trained vs coherent; curves for eval=0190 and eval=0198."""
    # {dr: {eval_tag: {ntrained: [coh,...]}}}
    buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for test_mod, train_mod, layer_t, dr, _frac, coh in _iter_results_txt_rows():
        if layer_t != ARCH_FILTER:
            continue
        if test_mod not in ("ML_F3W_WXIH0190", "ML_F3W_WXIH0198"):
            continue
        ntrained = _count_trained_modules(train_mod)
        buckets[dr][test_mod][ntrained].append(coh)

    for dr, eval_maps in sorted(buckets.items(), key=lambda kv: kv[0]):
        s0190_x = sorted(eval_maps.get("ML_F3W_WXIH0190", {}).keys())
        s0190_y = [float(np.mean(eval_maps["ML_F3W_WXIH0190"][n])) for n in s0190_x] if s0190_x else []
        s0198_x = sorted(eval_maps.get("ML_F3W_WXIH0198", {}).keys())
        s0198_y = [float(np.mean(eval_maps["ML_F3W_WXIH0198"][n])) for n in s0198_x] if s0198_x else []

        if not s0190_x and not s0198_x:
            continue

        y_all = []
        if s0190_y: y_all += s0190_y
        if s0198_y: y_all += s0198_y
        dy = 0.03 * (max(y_all) - min(y_all)) if len(y_all) >= 2 else 0.02
        xticks_vals = sorted(set(s0190_x) | set(s0198_x))

        os.makedirs(os.path.join("plots", "performance"), exist_ok=True)
        out_path = os.path.join("plots", "performance", f"coh_vs_numtrained_dr{dr}.pdf")

        plt.figure(figsize=(8, 6), dpi=300)
        if s0190_x:
            plt.plot(s0190_x, s0190_y, "-D", linewidth=2.0, markersize=7, label="ML_F3W_WXIH0190")
            for xi, yi in zip(s0190_x, s0190_y):
               plt.text(xi, yi + dy, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)
        if s0198_x:
            plt.plot(s0198_x, s0198_y, "-D", linewidth=2.0, markersize=7, label="ML_F3W_WXIH0198")
            for xi, yi in zip(s0198_x, s0198_y):
               plt.text(xi, yi - dy, f"{yi:.2f}", ha="center", va="top", fontsize=9)
        xticks_vals = sorted(set(s0190_x) | set(s0198_x))
        ax = plt.gca()
        ax.xaxis.set_major_locator(FixedLocator(xticks_vals))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        if xticks_vals:
            plt.xlim(min(xticks_vals) - 0.5, max(xticks_vals) + 0.5)

        if y_all:
            ymin, ymax = min(y_all), max(y_all)
            plt.ylim(ymin - 2*dy, ymax + 2*dy)

        plt.xlabel("# trained modules")
        plt.ylabel("Coherent noise ratio (corr/true)")
        plt.title(f"Train module count vs coherent noise ratio (dropout = {dr:g})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

# ----- Global plots all inputs and models included -----
def plot_and_save_graphs():
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 14,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "legend.frameon": True,
    })
    BLUE, ORANGE = "#1F77B4", "#FF7F0E"
    os.makedirs("plots/performance", exist_ok=True)

    def aggregate_results(data_dict):
        out = defaultdict(list)
        for (group_type, metric), values in data_dict.items():
            out[group_type].append((metric, np.mean(values)))
        return out

    # 1) Fractional RMS vs dropout (optional)
    if ENABLE_FRACTIONAL_PLOTS:
        dropout_frac = aggregate_results(dropout_group_results)
        plt.figure(figsize=(8, 6), dpi=300)
        for group_type, results in dropout_frac.items():
            results_sorted = sorted(results)
            x = [r[0] for r in results_sorted]
            y = [r[1] for r in results_sorted]
            plt.plot(x, y, '-D', color=BLUE if group_type == "SELF" else ORANGE, linewidth=2.2, markersize=8, label=group_type)
            for xi, yi in zip(x, y):
                plt.text(xi, yi + 0.01, f"{yi:.2f}", fontsize=10, ha='center')
        plt.xlabel('Dropout')
        plt.ylabel('Mean fractional RMS improvement')
        plt.title('Dropout vs fractional RMS improvement')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.savefig("plots/performance/dropout_fractional_rms_improvement.pdf")
        plt.close()

    # 2) Coherent ratio vs dropout
    dropout_coh = aggregate_results(dropout_coherent_group_results)
    plt.figure(figsize=(8, 6), dpi=300)
    for group_type, results in dropout_coh.items():
        results_sorted = sorted(results)
        x = [r[0] for r in results_sorted]
        y = [r[1] for r in results_sorted]
        plt.plot(x, y, '-D', color=BLUE if group_type == "SELF" else ORANGE, linewidth=2.2, markersize=8, label=group_type)
        for xi, yi in zip(x, y):
            plt.text(xi, yi + 0.03, f"{yi:.2f}", fontsize=10, ha='center')
    plt.xlabel('Dropout')
    plt.ylabel('Mean coherent ratio (corr/true)')
    plt.title('Dropout vs coherent noise ratio')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots/performance/dropout_coherent_noise_ratio.pdf")
    plt.close()

    # 3) Fractional RMS vs #train modules (optional)
    if ENABLE_FRACTIONAL_PLOTS:
        train_frac = aggregate_results(trained_group_results)
        plt.figure(figsize=(8, 6), dpi=300)
        for group_type, results in train_frac.items():
            results_sorted = sorted(results)
            x = [int(r[0]) for r in results_sorted]
            y = [r[1] for r in results_sorted]
            plt.plot(x, y, '-D', color=BLUE if group_type == "SELF" else ORANGE, linewidth=2.2, markersize=8, label=group_type)
            for xi, yi in zip(x, y):
                plt.text(xi, yi + 0.01, f"{yi:.2f}", fontsize=10, ha='center')
        plt.xlabel('# train modules')
        plt.ylabel('Mean fractional RMS improvement')
        plt.title('Train module count vs fractional RMS improvement')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xticks(sorted(set([int(r[0]) for results in train_frac.values() for r in results])))
        plt.tight_layout()
        plt.savefig("plots/performance/train_module_fractional_rms.pdf")
        plt.close()

    # 4) Coherent ratio vs #train modules
    train_coh = aggregate_results(trained_coherent_group_results)
    plt.figure(figsize=(8, 6), dpi=300)
    for group_type, results in train_coh.items():
        results_sorted = sorted(results)
        x = [int(r[0]) for r in results_sorted]
        y = [r[1] for r in results_sorted]
        plt.plot(x, y, '-D', color=BLUE if group_type == "SELF" else ORANGE, linewidth=2.2, markersize=8, label=group_type)
        for xi, yi in zip(x, y):
            plt.text(xi, yi + 0.03, f"{yi:.2f}", fontsize=10, ha='center')
    plt.xlabel('# train modules')
    plt.ylabel('Mean coherent ratio')
    plt.title('Train module count vs coherent noise ratio')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(sorted(set([int(r[0]) for results in train_coh.values() for r in results])))
    plt.tight_layout()
    plt.savefig("plots/performance/train_module_coherent_noise.pdf")
    plt.close()

# ----- Histograms-Validations -----
def plot_module_mean_histograms():
    """Module-level distributions."""
    os.makedirs("plots/performance", exist_ok=True)

    def plot_histogram(data_dict, title, xlabel, filename):
        means = [np.mean(vals) for vals in data_dict.values()]
        mean_of_means = np.mean(means)
        for module_name, vals in sorted(data_dict.items()):
            print(f"{title} | {module_name} → {float(np.mean(vals)):.4f}")
        plt.figure(figsize=(8,6), dpi=300)
        plt.hist(means, bins=10, color="#1F77B4", edgecolor='black', alpha=0.7)
        plt.axvline(mean_of_means, color='red', linestyle='--', label=f"mean={mean_of_means:.4f}")
        plt.xlabel(xlabel)
        plt.ylabel("Modules")
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"plots/performance/{filename}")
        plt.close()

    if ENABLE_FRACTIONAL_PLOTS:
        plot_histogram(
            per_module_fractional_means,
            "Per-module fractional RMS improvements",
            "Fractional RMS improvement",
            "histogram_fractional_module_means.pdf"
        )

    plot_histogram(
        per_module_coherent_means,
        "Per-module coherent noise ratios",
        "Coherent ratio (corr/true)",
        "histogram_coherent_module_means.pdf"
    )

def _plot_self_cross_hist_for_buckets(buckets, title_prefix, x_label, file_prefix):
    """SELF vs CROSS histograms per dropout + all."""
    os.makedirs("plots/performance", exist_ok=True)
    BLUE, ORANGE = "#1F77B4", "#FF7F0E"

    def _plot_one(drop_key, data_self, data_cross, suffix):
        all_vals = np.array(list(data_self) + list(data_cross))
        if len(all_vals) == 0:
            return
        bins = np.histogram_bin_edges(all_vals, bins="auto")
        plt.figure(figsize=(8, 6), dpi=300)
        if len(data_self) > 0:
            plt.hist(data_self, bins=bins, alpha=0.55, edgecolor="black", label=f"SELF (n={len(data_self)})", color=BLUE)
            plt.axvline(np.mean(data_self), linestyle="--", linewidth=1.8, color=BLUE, label=f"SELF mean = {np.mean(data_self):.4f}")
        if len(data_cross) > 0:
            plt.hist(data_cross, bins=bins, alpha=0.55, edgecolor="black", label=f"CROSS (n={len(data_cross)})", color=ORANGE)
            plt.axvline(np.mean(data_cross), linestyle="--", linewidth=1.8, color=ORANGE, label=f"CROSS mean = {np.mean(data_cross):.4f}")
        title = f"{title_prefix} – Dropout {drop_key}" if suffix else f"{title_prefix}"
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        out_name = f"plots/performance/{file_prefix}_{suffix}.pdf" if suffix else f"plots/performance/{file_prefix}_ALL.pdf"
        plt.tight_layout()
        plt.savefig(out_name)
        plt.close()

    for drop_key in sorted(buckets.keys()):
        dself = buckets[drop_key].get("SELF", [])
        dcross = buckets[drop_key].get("CROSS", [])
        _plot_one(drop_key, dself, dcross, f"dropout_{drop_key}")

    all_self, all_cross = [], []
    for d in buckets.values():
        all_self.extend(d.get("SELF", []))
        all_cross.extend(d.get("CROSS", []))
    _plot_one(None, all_self, all_cross, suffix=None)

def plot_self_cross_validation_histograms():
    """Top-level SELF/CROSS histograms."""
    if ENABLE_FRACTIONAL_PLOTS:
        frac_buckets = defaultdict(lambda: {"SELF": [], "CROSS": []})
        for (group_type, dropout_val), vals in dropout_group_results.items():
            frac_buckets[dropout_val][group_type].extend(vals)
        _plot_self_cross_hist_for_buckets(
            buckets=frac_buckets,
            title_prefix="Fractional RMS Improvement (SELF vs CROSS)",
            x_label="Fractional RMS improvement",
            file_prefix="hist_frac"
        )

    coh_buckets = defaultdict(lambda: {"SELF": [], "CROSS": []})
    for (group_type, dropout_val), vals in dropout_coherent_group_results.items():
        coh_buckets[dropout_val][group_type].extend(vals)
    _plot_self_cross_hist_for_buckets(
        buckets=coh_buckets,
        title_prefix="Coherent Ratio (SELF vs CROSS)",
        x_label="Coherent ratio (corr/true)",
        file_prefix="hist_coh"
    )

# ===== Compare channels: draw std curves per module from channel_stats.txt =====
_MODULE_LINE_RE = re.compile(r"^Module:\s*([^|]+)\s*\|\s*Train:\s*([^|]+)\s*\|\s*Model:\s*(.+)$")
_SECTION_RE = re.compile(r"^---\s*(TRUE|PRED)[^-\n]*---")
_ENTRY_RE = re.compile(r"^(ch_\d+|cm_erx\d+)\s+variance=([\-0-9.e+]+)\s+std=([\-0-9.e+]+)")

def _parse_channel_stats_std(stats_file: str):
    """
    Parse channel_stats.txt and collect per-module (TRUE, PRED) std arrays.
    Returns:
        true_map, pred_map, meta_map
        where meta_map[module] = {"train": "...", "dropout": 0.xx}
    """
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Stats file not found: {stats_file}")

    tmp = defaultdict(lambda: {"TRUE": {}, "PRED": {}})
    meta_map = {}

    current_module = None
    current_train = None
    current_model = None
    current_section = None

    with open(stats_file, "r") as f:
        for raw in f:
            line = raw.strip()

            m = _MODULE_LINE_RE.match(line)
            if m:
                current_module = m.group(1).strip()
                current_train = m.group(2).strip()
                current_model = m.group(3).strip()
                current_section = None

                # parse dropout from model string
                dr_match = re.search(r"__dr([0-9.]+)", current_model)
                dr_val = float(dr_match.group(1)) if dr_match else None
                meta_map[current_module] = {"train": current_train, "dropout": dr_val}
                continue

            s = _SECTION_RE.match(line)
            if s:
                current_section = s.group(1)
                continue

            e = _ENTRY_RE.match(line)
            if e and current_module and current_section in ("TRUE", "PRED"):
                name = e.group(1)
                if not name.startswith("ch_"):
                    continue
                ch_idx = int(name.split("_")[1])
                std_val = float(e.group(3))
                tmp[current_module][current_section][ch_idx] = std_val

    def _to_dense(section_key: str):
        dense = {}
        for mod, sec in tmp.items():
            idx2std = sec.get(section_key, {})
            if not idx2std:
                continue
            n = max(idx2std.keys()) + 1
            arr = np.full(n, np.nan, dtype=float)
            for i, v in idx2std.items():
                arr[i] = v
            dense[mod] = arr
        return dense

    return _to_dense("TRUE"), _to_dense("PRED"), meta_map

def overlay_hist(a, b, label_a, label_b, title, outfilename, bins=50):
    """
    Overlay histogram comparison of two arrays.
    Used for direct vs alternating channel sums.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    lo, hi  = min(a.min(), b.min()), max(a.max(), b.max())
    edges   = np.linspace(lo, hi, int(bins) + 1)
    plt.figure(figsize=(6,4))
    plt.hist(a, bins=edges, alpha=0.55, label=label_a)
    plt.hist(b, bins=edges, alpha=0.55, label=label_b)
    plt.title(title)
    plt.xlabel("sum (integrated ADC)")
    plt.ylabel("Event count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfilename)
    print(f"[overlay_hist] wrote plot {outfilename}")
    plt.close()

def plot_coherent_noise(y_true, y_pred, chadc, eventid, nch_per_erx, nerx, inputfolder, plotfolder, label_suffix=""):
    """
    Compute and plot coherent and incoherent noise before and after correction.
    Produces:
      - direct vs alternating sum histograms
      - 2D histograms of direct sums vs ERx index
      - 3-panel summary (incoherent/coherent, corr/true ratios, coh/inc ratios)
    """
    coh_true_list, inc_true_list = [], []
    coh_corr_list, inc_corr_list = [], []
    all_dir_true_per_erx, all_alt_true_per_erx = [], []
    all_dir_corr_per_erx, all_alt_corr_per_erx = [], []

    for erx in range(nerx):
        order = np.lexsort((chadc, eventid))  # sort by event then channel
        chadc_ordered   = chadc[order]
        eventid_ordered = eventid[order]
        y_true_ordered  = y_true[order]
        y_pred_ordered  = y_pred[order]

        # Select only this ERx board’s channels
        mask  = (chadc_ordered >= erx * nch_per_erx) & (chadc_ordered < (erx + 1) * nch_per_erx)
        idxs  = np.where(mask)[0]

        adc_true = y_true_ordered[idxs]
        adc_pred = y_pred_ordered[idxs]
        adc_corr = adc_true - adc_pred

        # Reshape to (n_events, nch_per_erx)
        n_rows = adc_true.size
        assert n_rows % nch_per_erx == 0, "rows not multiple of nch_per_erx"
        n_evt  = n_rows // nch_per_erx

        adc_true_2d = adc_true.reshape(n_evt, nch_per_erx)
        adc_corr_2d = adc_corr.reshape(n_evt, nch_per_erx)

        # Direct and alternating sums per event
        dir_sums_true = adc_true_2d.sum(axis=1)
        alt_sums_true = adc_true_2d[:, ::2].sum(axis=1) - adc_true_2d[:, 1::2].sum(axis=1)

        dir_sums_corr = adc_corr_2d.sum(axis=1)
        alt_sums_corr = adc_corr_2d[:, ::2].sum(axis=1) - adc_corr_2d[:, 1::2].sum(axis=1)

        all_dir_true_per_erx.append(dir_sums_true)
        all_alt_true_per_erx.append(alt_sums_true)
        all_dir_corr_per_erx.append(dir_sums_corr)
        all_alt_corr_per_erx.append(alt_sums_corr)

        # Compute incoherent and coherent noise estimates
        delta_true = dir_sums_true.var() - alt_sums_true.var()
        delta_corr = dir_sums_corr.var() - alt_sums_corr.var()

        inc_noise_true = np.std(alt_sums_true) / math.sqrt(nch_per_erx)
        coh_noise_true = np.sign(delta_true) * np.sqrt(abs(delta_true)) / nch_per_erx

        inc_noise_corr = np.std(alt_sums_corr) / math.sqrt(nch_per_erx)
        coh_noise_corr = np.sign(delta_corr) * np.sqrt(abs(delta_corr)) / nch_per_erx

        coh_true_list.append(coh_noise_true)
        inc_true_list.append(inc_noise_true)
        coh_corr_list.append(coh_noise_corr)
        inc_corr_list.append(inc_noise_corr)

        """
        # Plot overlay histograms of direct vs alternating sums
        overlay_hist(dir_sums_true, alt_sums_true,
                     "direct sum channels", "alternating sum channels",
                     "Uncorrected per-event sums",
                     outfilename=f"{os.path.join(plotfolder, 'sums_true')}_erx{erx:02d}.pdf")
        overlay_hist(dir_sums_corr, alt_sums_corr,
                     "direct sum channels (corrected)", "alternating sum channels (corrected)",
                     "Corrected per-event sums",
                     outfilename=f"{os.path.join(plotfolder, 'sums_corrected')}_erx{erx:02d}.pdf")
        """
    # Convert lists to arrays
    coh_true_arr = np.array(coh_true_list)
    inc_true_arr = np.array(inc_true_list)
    coh_corr_arr = np.array(coh_corr_list)
    inc_corr_arr = np.array(inc_corr_list)
    erx_idx      = np.arange(nerx)

    # Build 2D histogram of direct sums vs ERx
    n_bins = 50
    all_dir_flat = np.concatenate(all_dir_true_per_erx)
    y_min, y_max = all_dir_flat.min(), all_dir_flat.max()
    bin_edges = np.linspace(y_min, y_max, n_bins + 1)

    hist2d = np.zeros((n_bins, nerx), dtype=int)
    for erx, vec in enumerate(all_dir_true_per_erx):
        hist2d[:, erx], _ = np.histogram(vec, bins=bin_edges)

    fig, ax = plt.subplots(figsize=(7,5))
    extent = [-0.5, nerx-0.5, y_min, y_max]
    im = ax.imshow(hist2d, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    ax.set_xlabel("ERx board index")
    ax.set_xticks(erx_idx)
    ax.set_ylabel("Direct sum (ADC)")
    cb = fig.colorbar(im, ax=ax); cb.set_label("Events")
    plt.tight_layout()
    plt.savefig(f"{plotfolder}/ds_true_vs_erx.pdf")

    # Ratios
    inc_ratio     = inc_corr_arr / inc_true_arr
    coh_ratio     = coh_corr_arr / coh_true_arr
    coh_inc_true  = coh_true_arr / inc_true_arr
    coh_inc_corr  = coh_corr_arr / inc_corr_arr

    # 3-panel figure
    fig = plt.figure(figsize=(7, 6))
    gs  = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)

    ax1 = fig.add_subplot(gs[0])   # main panel
    axr = fig.add_subplot(gs[1], sharex=ax1)   # ratio panel
    axc = fig.add_subplot(gs[2], sharex=ax1)   # coh/inc panel

    # Main panel
    ax1.plot(erx_idx, inc_true_arr,  "o-",  label="incoherent (true)", color="tab:blue")
    ax1.plot(erx_idx, coh_true_arr,  "s-",  label="coherent (true)",   color="tab:orange")
    ax1.plot(erx_idx, inc_corr_arr, "o--", label="incoherent (corr)", color="tab:blue")
    ax1.plot(erx_idx, coh_corr_arr, "s--", label="coherent (corr)",   color="tab:orange")
    ax1.set_ylabel("Noise (ADC)", fontsize=16, loc="top", labelpad=12)
    ax1.set_ylim(0., 3.)
    ax1.grid(ls="--", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=14)

    # Ratio panel
    axr.plot(erx_idx, inc_ratio, "o--", color="tab:blue")
    axr.plot(erx_idx, coh_ratio, "s--", color="tab:orange")
    axr.set_xlabel("e-Rx", fontsize=16, loc="right", labelpad=8)
    axr.set_ylabel("corr / true", fontsize=14, loc="top", labelpad=12)
    axr.set_ylim(0., 1.1)
    axr.grid(ls="--", alpha=0.3)

    # coh/inc panel
    axc.plot(erx_idx, coh_inc_true, "D-",  color="black")
    axc.plot(erx_idx, coh_inc_corr, "D--", color="black")
    axc.set_xlabel("e-Rx", fontsize=16, loc="right", labelpad=8)
    axc.set_ylabel("coh / inc", fontsize=14, loc="top", labelpad=8)
    axc.set_ylim(0., 2.)
    axc.grid(ls="--", alpha=0.3)

    # Hide x-ticks on upper two panels
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(axr.get_xticklabels(), visible=False)

    plt.tight_layout()
    outfilename = os.path.join(plotfolder, "noise_fractions_with_ratio.pdf")
    if label_suffix:
        outfilename = outfilename.replace(".pdf", f"_{label_suffix}.pdf")
    fig.savefig(outfilename, bbox_inches="tight", pad_inches=0.05)

def _run_noise_fraction_plots_for_selected(base_dir=BASE_DIR):
    """
    Iterate over all bundles and produce coherent/incoherent noise plots.
    Respects ENABLE_NOISE_FRACTIONS_PLOT, NOISE_FRACTIONS_MODULES, and ARCH_FILTER.
    """
    if not ENABLE_NOISE_FRACTIONS_PLOT:
        print("[info] Noise fractions plot disabled.")
        return

    selected = set(NOISE_FRACTIONS_MODULES) if NOISE_FRACTIONS_MODULES else None

    for emod, tmod, model, bundle in _iter_all_bundles(base=base_dir):

        # Filter by module
        if selected is not None and emod not in selected:
            continue

        # Filter by architecture string
        if ARCH_FILTER is not None:
            arch_str = f"in20__{'-'.join(map(str, ARCH_FILTER))}"
            if arch_str not in model:
                continue

        plotfolder = os.path.dirname(bundle)
        try:
            with gzip.open(bundle, "rb") as f:
                b = pickle.load(f)
        except Exception as e:
            print(f"[warn] Cannot open bundle: {bundle} ({e})")
            continue

        meta    = b.get("meta", {})
        frames  = b.get("frames", {})
        indices = b.get("indices", {})

        if not frames or not indices or not meta:
            print(f"[warn] Missing keys in bundle: {bundle}")
            continue

        # Extract arrays
        # Build arrays directly from frames (event × channel order)
        true_df = frames["true"]
        pred_df = frames["pred"]

        # Select only channel columns to ensure correct order
        ch_cols = [c for c in true_df.columns if str(c).startswith("ch_")]
        true_df = true_df[ch_cols]
        pred_df = pred_df[ch_cols]

        # Flatten y_true / y_pred in event-major order (row by row)
        y_true = true_df.to_numpy().ravel(order="C")
        y_pred = pred_df.to_numpy().ravel(order="C")

        # Generate matching eventid / chadc arrays in the same order
        ev_ids = true_df.index.to_numpy()
        nch = len(ch_cols)
        eventid = np.repeat(ev_ids, nch)

        # Parse channel numbers from column names to build chadc
        ch_nums = np.array([int(c.split("_")[1]) for c in ch_cols], dtype=int)
        chadc = np.tile(ch_nums, len(ev_ids))

        # Safety check
        assert y_true.shape == y_pred.shape == eventid.shape == chadc.shape, "Shape mismatch!"

        nch_per  = int(meta.get("nch_per_erx", 37))
        nerx     = int(meta.get("nerx", 6))

        os.makedirs(plotfolder, exist_ok=True)
        print(f"[coh/inc plot] eval={emod} | train={tmod} | model={model}")

        # Run plotting
        plot_coherent_noise(
            y_true=y_true,
            y_pred=y_pred,
            chadc=chadc,
            eventid=eventid,
            nch_per_erx=nch_per,
            nerx=nerx,
            inputfolder="",
            plotfolder=plotfolder,
            label_suffix=f"{emod}"
        )



def overlay_profiles_multi(xs_list, ys_list, labels_profiles, label_x, label_y, output_filename, nbins_x=50):
    """
    Draw overlay profiles for multiple modules.
    Each series can have its own X and Y.
    Legend is placed outside the plot area.
    """

    # Safety checks
    if not (len(xs_list) == len(ys_list) == len(labels_profiles)):
        raise ValueError("Lengths of xs_list, ys_list, and labels_profiles must match")

    # Define common binning
    x_min = min(np.min(x) for x in xs_list if x.size > 0)
    x_max = max(np.max(x) for x in xs_list if x.size > 0)
    bins = np.linspace(x_min, x_max, nbins_x + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    # Compute means
    prof_means = []
    for x, y in zip(xs_list, ys_list):
        idx = np.digitize(x, bins) - 1
        m = np.full(nbins_x, np.nan, dtype=float)
        for i in range(nbins_x):
            mask = idx == i
            if np.any(mask):
                vals = y[mask]
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    m[i] = np.mean(vals)
        prof_means.append(m)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for m, lab in zip(prof_means, labels_profiles):
        ax.plot(centers, m, label=lab, linewidth=1.8)

    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.grid(True, alpha=0.3)

    # Legend outside plot
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 0.8, 1])  # leave space for legend
    fig.savefig(output_filename)
    plt.close(fig)

def plot_module_cm_profiles_dr0(eval_module: str,
                                base_dir: str = BASE_DIR,
                                out_root: str = os.path.join("plots", "performance", "compare_channels", "global_cm_profiles")):
    """
    For a given evaluation module:
      - Iterate over all bundles and select only models with dropout=0.
      - Load the TRUE series once for this module.
      - For each CM column: prepare x=CM (with channel repeats),
        y=[TRUE, all dr=0 DNN predictions].
      - Plot with utils.overlay_profiles (same as profile_cm) and save to PDF.
    
    Output:
      plots/performance/compare_channels/global_cm_profiles/<EVAL_MODULE>/<CM_NAME>.pdf

    Legend entries:
      - 'true'
      - 'DNN: <train_module>/<model_name>'
    """

    bundles = []
    first_true = None

    # --- Iterate over all bundles for this module ---
    for emod, tmod, model, bundle in _iter_all_bundles(base=base_dir):
        if emod != eval_module:
            continue
        try:
            with gzip.open(bundle, "rb") as f:
                b = pickle.load(f)
        except Exception:
            continue

        meta   = b.get("meta", {})
        frames = b.get("frames", {})
        if not meta or not frames:
            continue

        # Get TRUE data once
        if first_true is None:
            cm_df   = frames.get("cm")
            true_df = frames.get("true")
            if cm_df is None or true_df is None:
                continue
            n_channels = true_df.shape[1]
            x_by_cm = {
                c: np.repeat(cm_df[c].to_numpy(), n_channels)
                for c in cm_df.columns
            }
            y_true_flat = true_df.to_numpy().ravel()
            first_true = (x_by_cm, y_true_flat)

        # Select only dropout=0 models
        nodes_per_layer, dr = parse_model_config(meta.get("model_name", ""))
        if nodes_per_layer is None or dr is None or abs(dr - 0.0) > 1e-12:
            continue
        bundles.append((meta, frames))

    if first_true is None:
        print(f"[profiles] {eval_module}: TRUE/CM data not found.")
        return
    if not bundles:
        print(f"[profiles] {eval_module}: no dr=0 DNN models found, skipped.")
        return

    x_by_cm, y_true_flat = first_true
    mod_out_dir = os.path.join(out_root, _sanitize(eval_module))
    os.makedirs(mod_out_dir, exist_ok=True)

    # --- Plot per CM column ---
    for cm_name, x_flat in x_by_cm.items():
        y_list = [y_true_flat]
        labels = ["true"]

        for meta, frames in bundles:
            y_pred_flat = frames.get("pred").to_numpy().ravel()

            # Robust count of trained modules from the train_module string
            train_module_str = meta.get("train_module", "")
            if train_module_str in EXCLUDE_TRAIN_MODULES:
                continue
            if train_module_str:
                n_train = len(re.findall(r"ML_[A-Z0-9_]+?(?=_ML_|$)", train_module_str))
            else:
                train_modules = meta.get("train_modules", ["UNKNOWN"])
                n_train = len(train_modules)

            # Legend label based on module count
            if n_train == 1:
                labels.append("DNN: 1 module")
            else:
                labels.append(f"DNN: {n_train} modules")

            y_list.append(y_pred_flat)


        out_path = os.path.join(mod_out_dir, f"profiles__{_sanitize(eval_module)}__{_sanitize(cm_name)}.pdf")
        # Use a temporary rcContext so only these plots get a smaller legend
        with mpl.rc_context({"legend.fontsize": "small"}):
            utils.overlay_profiles(
                vals_x=x_flat,
                list_of_vals_y=y_list,
                label_x=f"{cm_name} (ADC)",
                label_y="ADC",
                labels_profiles=labels,
                output_filename=out_path,
                nbins_x=50
            )
        print(f"[ok] Saved → {out_path}")



def aggregate_and_plot_cm_profiles_with_modules(selected_train_module: str,
                                                selected_dropout: float,
                                                base_dir: str = BASE_DIR):
    arch_str = f"in20__{'-'.join(map(str, ARCH_FILTER))}" if ARCH_FILTER else None
    if arch_str and arch_str not in model:   # model string or bundle path
        return
    """
    For the given train_module and dropout:
    - Collect all eval modules' bundles
    - For each CM column, plot three graphs:
        (1) TRUE-only (all modules, legend=module name)
        (2) DNN-only  (all modules, legend=module name)
        (3) TRUE+DNN overlay (legend='module true', 'module dnn')
    """
    from collections import defaultdict

    true_by_cm = defaultdict(dict)
    pred_by_cm = defaultdict(dict)
    x_by_cm    = defaultdict(dict)

    for emod, tmod, model, bundle in _iter_all_bundles(base=base_dir):
        try:
            with gzip.open(bundle, "rb") as f:
                b = pickle.load(f)
        except Exception:
            continue

        meta   = b.get("meta", {})
        frames = b.get("frames", {})
        if not meta or not frames:
            continue
        if str(meta.get("train_module", "")) != str(selected_train_module):
            continue
        dr = float(meta.get("dropout_rate", float("nan")))
        if not np.isfinite(dr) or abs(dr - float(selected_dropout)) > 1e-9:
            continue

        cm_df   = frames.get("cm")
        true_df = frames.get("true")
        pred_df = frames.get("pred")
        if cm_df is None or true_df is None or pred_df is None:
            continue

        n_channels = true_df.shape[1]
        y_true_flat = true_df.to_numpy().ravel()
        y_pred_flat = pred_df.to_numpy().ravel()

        for cm_name in cm_df.columns:
            x_flat = np.repeat(cm_df[cm_name].to_numpy(), n_channels)
            x_by_cm[cm_name][emod]    = x_flat
            true_by_cm[cm_name][emod] = y_true_flat
            pred_by_cm[cm_name][emod] = y_pred_flat

    if not x_by_cm:
        print(f"[aggregate] No bundles matched for train='{selected_train_module}', dropout={selected_dropout}")
        return

    safe_train = _sanitize(selected_train_module)
    out_root = os.path.join("plots", "performance", "compare_channels",
                            "global_cm_profiles", safe_train, f"dr_{selected_dropout:g}")
    os.makedirs(out_root, exist_ok=True)

    for cm_name in sorted(x_by_cm.keys()):
        prof_dir = os.path.join(out_root, _sanitize(cm_name))
        os.makedirs(prof_dir, exist_ok=True)

        modules = sorted(true_by_cm[cm_name].keys())

        # TRUE-only
        xs_true   = [x_by_cm[cm_name][m] for m in modules]
        ys_true   = [true_by_cm[cm_name][m] for m in modules]
        overlay_profiles_multi(xs_true, ys_true, modules,
                               label_x=f"{cm_name} (ADC)", label_y="ADC",
                               output_filename=os.path.join(prof_dir, f"{_sanitize(cm_name)}__true_only.pdf"))

        # DNN-only
        xs_dnn   = [x_by_cm[cm_name][m] for m in modules]
        ys_dnn   = [pred_by_cm[cm_name][m] for m in modules]
        overlay_profiles_multi(xs_dnn, ys_dnn, modules,
                               label_x=f"{cm_name} (ADC)", label_y="ADC",
                               output_filename=os.path.join(prof_dir, f"{_sanitize(cm_name)}__dnn_only.pdf"))

        # TRUE+DNN overlay
        xs_mix, ys_mix, labels_mix = [], [], []
        for m in modules:
            xs_mix.append(x_by_cm[cm_name][m]); ys_mix.append(true_by_cm[cm_name][m]); labels_mix.append(f"{m} true")
            xs_mix.append(x_by_cm[cm_name][m]); ys_mix.append(pred_by_cm[cm_name][m]); labels_mix.append(f"{m} dnn")

        overlay_profiles_multi(xs_mix, ys_mix, labels_mix,
                               label_x=f"{cm_name} (ADC)", label_y="ADC",
                               output_filename=os.path.join(prof_dir, f"{_sanitize(cm_name)}__true_vs_dnn.pdf"))

    print(f"[aggregate] Saved global CM profiles with modules → {out_root}")



# ----- State -----
def reset_state():
    layer_results.clear()
    detailed_results.clear()
    coherent_results.clear()
    coherent_detailed_results.clear()
    dropout_group_results.clear()
    dropout_coherent_group_results.clear()
    per_module_fractional_means.clear()
    per_module_coherent_means.clear()
    existing_result_keys.clear()



# ===== Main =====
if __name__ == '__main__':
    reset_state()
    load_existing_results()
    os.environ["USER"] = "areimers"
    username_load_model_from = "areimers"
    modules, models_dict = discover_modules_and_models(username_load_model_from)
    modules_to_process = filter_existing_plots(modules, models_dict)

    print("Modules to process:")
    for test_module, combo_list in modules_to_process.items():
        for train_module, model_name in combo_list:
            print(f"   - Evaluate Module: {test_module} | Train: {train_module} | Model: {model_name}")
    print("\nStarting evaluation...\n")

    for test_module, combo_list in modules_to_process.items():
        for train_module, model_name in combo_list:
            # --- Step 0: Filters ---
            if not model_name.startswith("in20"):
                print(f"Skipping {model_name} (not starting with 'in20')")
                continue
            nodes_per_layer, dropout_rate = parse_model_config(model_name)
            if nodes_per_layer is None or dropout_rate is None:
                print(f"Skipping model {model_name} (invalid format)")
                continue

            key = (test_module, train_module, tuple(nodes_per_layer), float(dropout_rate))
            plotfolder  = f"plots/performance/{test_module}/{train_module}/{model_name}"
            bundle_path = os.path.join(plotfolder, "predictions_bundle.pkl.gz")

            # --- Step 1: Skip if entry already exists in results.txt ---
            if key in existing_result_keys:
                print(f"[skip] results.txt already has entry for {key}")
                continue

            # --- Step 2: Evaluate and write results.txt ---
            bundle_exists_before = os.path.exists(bundle_path)
            print(f"[evaluate] Eval={test_module} | Train={train_module} | Model={model_name} | bundle_exists={bundle_exists_before}")

            res = evaluate_model_and_compute_metrics(
                modulename_for_evaluation=test_module,
                train_module=train_module,
                model_name=model_name,
                username_load_model_from=username_load_model_from,
                write_bundle=not bundle_exists_before,  # do not overwrite if bundle already exists
                write_plots=False                      # always draw plots later from bundle
            )

            # If the module was empty/missing, we get None → skip everything for this combo.
            if res is None:
                continue

            frac_impr_mean, coh_ratio_mean = res

            register_result(
                test_module, train_module, nodes_per_layer, dropout_rate,
                frac_impr_mean, coh_ratio_mean
            )

            # --- Step 3: Always plot from bundle ---
            bundle_exists_after = os.path.exists(bundle_path)
            if bundle_exists_before or bundle_exists_after:
                print(f"[plots] Using bundle → {bundle_path}")

                if GENERATE_CM_PLOTS:
                    # Cov/Corr plot from bundle
                    plot_covcorr_from_bundle(bundle_path, out_dir=plotfolder)

                    # --- CM vs ADC scatter with marginals ---
                    with gzip.open(bundle_path, "rb") as f:
                        b = pickle.load(f)

                    frames = b["frames"]
                    cm_df  = frames["cm"]
                    y_true = frames["true"]

                    for cm_name in cm_df.columns:
                        # repeat CM per event across all channels to align with flattened ADCs
                        x_flat = np.repeat(cm_df[cm_name].to_numpy(), y_true.shape[1])
                        y_true_flat = y_true.to_numpy().ravel()

                        subdir = os.path.join(plotfolder, "cm_scatter")
                        os.makedirs(subdir, exist_ok=True)

                        utils.plot_y_vs_x_with_marginals(
                            vals_x=x_flat,
                            vals_y=y_true_flat,
                            label_x=f"{cm_name} (ADC)",
                            label_y="Uncorrected (ADC)",
                            label_profile="profile",
                            output_filename=os.path.join(subdir, f"uncorr_vs_{cm_name}.pdf"),
                            nbins_x=80,
                            nbins_y=80
                        )

                        # --- Overlaid profiles: TRUE vs DNN & residuals ---
                        y_pred_flat = frames["pred"].to_numpy().ravel()
                        y_res_flat  = frames["residual"].to_numpy().ravel()

                        prof_dir = os.path.join(plotfolder, "cm_profiles")
                        os.makedirs(prof_dir, exist_ok=True)

                        utils.overlay_profiles(
                            vals_x=x_flat,
                            list_of_vals_y=[y_true_flat, y_pred_flat],
                            label_x=f"{cm_name} (ADC)",
                            label_y="ADC",
                            labels_profiles=["true", "dnn"],
                            output_filename=os.path.join(prof_dir, f"profiles_variants_{cm_name}.pdf"),
                            nbins_x=50,
                            ratio_to=0
                        )

                        utils.overlay_profiles(
                            vals_x=x_flat,
                            list_of_vals_y=[y_res_flat],
                            label_x=f"{cm_name} (ADC)",
                            label_y="ADC",
                            labels_profiles=["dnn residual"],
                            output_filename=os.path.join(prof_dir, f"profiles_residuals_{cm_name}.pdf"),
                            nbins_x=50
                        )

            else:
                print("[warn] Bundle not found; ensure evaluate() writes it when bundle is missing.")


    print("\n=====  Fractional RMS Improvement Summary =====")
    for (evaluate_module, train_module, layer_key), values in layer_results.items():
        print(f"🔎 Eval {evaluate_module} | Train {train_module} | Layer {layer_key} → Mean FRACT = {np.mean(values):.4f}")

    print("\n=====  Detailed Fractional RMS (by dropout) =====")
    for (evaluate_module, train_module, layer_key), values in detailed_results.items():
        print(f"\n Eval: {evaluate_module} | Train: {train_module} | Layer: {layer_key}")
        for dr_value, mean_val in values:
            print(f"  • Dropout {dr_value}: {mean_val:.4f}")

    print("\n=====  Dropout-Based Mean Fractional RMS Improvement =====")
    grouped_results = defaultdict(list)
    for (group_type, dropout_val), values in dropout_group_results.items():
        grouped_results[group_type].append((dropout_val, np.mean(values)))
    for group_type, dropout_list in grouped_results.items():
        print(f"\n {group_type} Models:")
        for dropout_val, mean_val in sorted(dropout_list):
            print(f"  • Dropout {dropout_val}: {mean_val:.4f}")

    print("\n=====  Dropout-Based Mean Coherent Noise Ratio (corr/true) =====")
    coh_grouped_results = defaultdict(list)
    for (group_type, dropout_val), values in dropout_coherent_group_results.items():
        coh_grouped_results[group_type].append((dropout_val, np.mean(values)))
    for group_type, dropout_list in coh_grouped_results.items():
        for dropout_val, mean_val in sorted(dropout_list):
            print(f"{group_type} | Dropout {dropout_val}: Mean Coherent = {mean_val:.4f}")

    # Train-module count groupings
    trained_group_results = defaultdict(list)
    trained_coherent_group_results = defaultdict(list)

    def count_trained_modules(train_module_name):
        return len(re.findall(r'ML_[A-Z0-9]+', train_module_name))

    print("Layer results content:")
    for k, v in layer_results.items():
        print(k, v)

    for (evaluate_module, train_module, layer_key), values in layer_results.items():
        num_trained = count_trained_modules(train_module)
        group_type = "SELF" if is_self(evaluate_module, train_module) else "CROSS"
        trained_group_results[(group_type, num_trained)].extend(values)
        trained_coherent_group_results[(group_type, num_trained)].extend(coherent_results[(evaluate_module, train_module, layer_key)])

    print("\n=====  Train Module Count-Based Mean Fractional RMS Improvement =====")
    for (group, n), vals in trained_group_results.items():
        print(f"{group} | Train Modules = {n}: Mean Fractional RMS = {np.mean(vals):.4f}")

    print("\n=====  Train Module Count-Based Mean Coherent Noise Ratio =====")
    for (group, n), vals in trained_coherent_group_results.items():
        print(f"{group} | Train Modules = {n}: Mean Coherent Noise = {np.mean(vals):.4f}")

    # Plots-global
    plot_and_save_graphs()
    plot_module_mean_histograms()

    # Plots for module 098 090
    plot_new_dropout_vs_coherent_by_train()
    plot_new_numtrained_vs_coherent_by_dropout()
    #Validation
    plot_self_cross_validation_histograms()

    # === Run global cm profiles (only for selected modules) ===
    if RUN_GLOBAL_CM_PROFILES:
        try:
            for em in MODULES_FOR_GLOBAL_CM_PROFILES:
                plot_module_cm_profiles_dr0(em, base_dir=BASE_DIR)
        except Exception as e:
            print(f"[warn] global_cm_profiles generation failed: {e}")
    else:
        print("[info] global_cm_profiles generation skipped (RUN_GLOBAL_CM_PROFILES=False).")


    _run_noise_fraction_plots_for_selected(base_dir=BASE_DIR)

    # --- Final info about CM plotting ---
    if not GENERATE_CM_PLOTS:
        print("[info] CM/cov-corr plotting disabled globally (GENERATE_CM_PLOTS=False)")


    run_global_covcorr()
    run_analytic_jobs(username_load_model_from="areimers")

    # === Run analytic noise plots (from analytic bundles) ===
    run_analytic_noise_fraction_plots(base_dir=BASE_DIR)

    # --- Analytic global cov/corr (k=0) ---
    run_analytic_global_covcorr()

    # =======  Projection histograms =========
    run_projection_hists()

    if ENABLE_DISTCORR_PLOTS:
        run_distance_corr_plots()

    if ENABLE_MULTIPLE_INPUT_EXPORT:
        run_multiple_input_export()

    if ENABLE_MULTI_INPUT_DNN_INFERENCE:
        run_multi_input_dnn_inference(username_load_model_from="areimers")


    run_build_multi_inputs_from_modules()


    run_multi_dnn_postprocess_only()
