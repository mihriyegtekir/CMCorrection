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
RUN_GLOBAL_CM_PROFILES = True   # Set to False to skip this part entirely
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
ENABLE_NOISE_FRACTIONS_PLOT = True         # Enable/disable coherent/incoherent noise plots
NOISE_FRACTIONS_MODULES = ["ML_F3W_WXIH0190", "ML_F3W_WXIH0191" "ML_F3W_WXIH0194", "ML_F3W_WXIH0197", "ML_F3W_WXIH0198"]               # [] → all eval modules, or specify list of module names
ARCH_FILTER = (512, 512, 512, 512, 64)     # Only run for models with this architecture; set to None to disable



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
    """Parse nodes and dropout from folder name."""
    nodes = re.search(r"__(\d+(?:-\d+)+)__", model_name)
    if not nodes:
        return None, None
    nodes_per_layer = [int(n) for n in nodes.group(1).split("-")]
    dr = re.search(r"__dr([0-9.]+)", model_name)
    if not dr:
        return None, None
    return nodes_per_layer, float(dr.group(1))

_MODULE_RE = re.compile(r"ML_[A-Z0-9_]+?(?=_ML_|$)")

def is_self(test_module: str, train_module: str) -> bool:
    """SELF if test module is included in training name."""
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

def filter_existing_plots(modules, models, output_base="plots/performance"):
    """Skip combos already logged (and/or with existing folders)."""
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
        import numpy as np
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


    with open(f"{inputfolder}/colnames.json") as f:
        colnames_inputs = json.load(f)

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
    """
    print(f"[DEBUG CODE2] TRUE cov shape={cov_true.shape}, corr shape={corr_true.shape}")
    print(cov_true.iloc[:3, :3].round(4))
    print(corr_true.iloc[:3, :3].round(4))

    print(f"[DEBUG CODE2] DNN cov shape={cov_pred.shape}, corr shape={corr_pred.shape}")
    print(cov_pred.iloc[:3, :3].round(4))
    print(corr_pred.iloc[:3, :3].round(4))

    print(f"[DEBUG CODE2] RESIDUAL cov shape={cov_res.shape}, corr shape={corr_res.shape}")
    print(cov_res.iloc[:3, :3].round(4))
    print(corr_res.iloc[:3, :3].round(4))
    """
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
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))  # 1.0 yerine 1 yazsın
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
    import numpy as np, matplotlib.pyplot as plt, os

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
    import os, gzip, pickle, numpy as np

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
    import os, numpy as np, gzip, pickle
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

            frac_impr_mean, coh_ratio_mean = evaluate_model_and_compute_metrics(
                modulename_for_evaluation=test_module,
                train_module=train_module,
                model_name=model_name,
                username_load_model_from=username_load_model_from,
                write_bundle=not bundle_exists_before,  # do not overwrite if bundle already exists
                write_plots=False                      # always draw plots later from bundle
            )

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
