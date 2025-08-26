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

# ===== Config =====
ENABLE_RESULTS_TXT_SKIP = True
ENABLE_FOLDER_SKIP = False
GENERATE_DIAGNOSTIC_PLOTS = False  # validation histograms
ENABLE_FRACTIONAL_PLOTS = False    # fractional RMS plots
RESULTS_FILE = "plots/performance/results.txt"
ARCH_FILTER = (512, 512, 512, 512, 64)  # only in20__512-512-512-512-64*
APPLY_ARCH_FILTER_TO_AGGREGATES = True  # apply layer filter to ALL existing charts

def _arch_ok(layer_tuple):
    """Return True if row passes ARCH filter for aggregates."""
    return (not APPLY_ARCH_FILTER_TO_AGGREGATES) or (layer_tuple == ARCH_FILTER)


existing_result_keys = set()

def _sanitize(name: str) -> str:
    """Filesystem-safe: keep letters, numbers, _ and -; collapse others to '_'."""
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', name).strip('_')

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

def evaluate_model_and_compute_metrics(modulename_for_evaluation: str, train_module: str, model_name: str, username_load_model_from: str):
    """Load model, run inference, compute (frac_impr_mean, coh_ratio_mean)."""
    device = torch.device("cpu")
    nodes_per_layer, dropout_rate = parse_model_config(model_name)
    assert nodes_per_layer and dropout_rate is not None, f"Invalid model name: {model_name}"

    nch_per_erx, nerx = _infer_erx_params(modulename_for_evaluation)

    inputfolder = f"/eos/user/{os.getenv('USER')[0]}/{os.getenv('USER')}/hgcal/dnn_inputs/{modulename_for_evaluation}"
    modelfolder = f"/eos/user/{username_load_model_from[0]}/{username_load_model_from}/hgcal/dnn_models/{train_module}/{model_name}"
    plotfolder  = f"plots/performance/{modulename_for_evaluation}/{train_module}/{model_name}"
    os.makedirs(plotfolder, exist_ok=True)

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
    frac_impr       = 1.0 - (rms_corrected_per_chan / rms_true_per_channel)
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
def plot_dropout_vs_coherent_by_train():
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

def plot_numtrained_vs_coherent_by_dropout():
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
            if not model_name.startswith("in20"):
                print(f"Skipping {model_name} (not starting with 'in20')")
                continue
            print(f"Running evaluation | Eval: {test_module} | Train: {train_module} | Model: {model_name}")

            nodes_per_layer, dropout_rate = parse_model_config(model_name)
            if nodes_per_layer is None or dropout_rate is None:
                print(f"Skipping model {model_name} (invalid format)")
                continue

            frac_impr_mean, coh_ratio_mean = evaluate_model_and_compute_metrics(
                modulename_for_evaluation=test_module,
                train_module=train_module,
                model_name=model_name,
                username_load_model_from=username_load_model_from
            )

            register_result(test_module, train_module, nodes_per_layer, dropout_rate, frac_impr_mean, coh_ratio_mean)

        if test_module in per_module_fractional_means:
            print(f"\n Finished {test_module} → Mean Fractional RMS Improvement: {np.mean(per_module_fractional_means[test_module]):.4f}")
        if test_module in per_module_coherent_means:
            print(f" Finished {test_module} → Mean Coherent Noise Ratio: {np.mean(per_module_coherent_means[test_module]):.4f}")

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
    plot_dropout_vs_coherent_by_train()
    plot_numtrained_vs_coherent_by_dropout()
    #Validation
    plot_self_cross_validation_histograms()
