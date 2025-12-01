import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
import evaluate_performance as ep  # Import analytic & plotting utilities
from dataclasses import dataclass, field
import torch
from models import DNNFlex
from evaluate_performance import DNNInferencer
from typing import Union, List, Optional
import matplotlib
matplotlib.use("Agg")           
import matplotlib.pyplot as plt
import time


# ============================================================
# Configuration Class
# ============================================================

@dataclass
class CompareConfig:
    """
    Global configuration for the compare_methods pipeline.
    Defines paths, behavior flags, and analytic model parameters.
    """
    base_input_dir: str = "/eos/user/a/areimers/hgcal/dnn_inputs"
    base_output_dir: str = "./method_A"
    plots_output_dir: str = "./plots/performance"
    reference_module: str = "ML_F3W_WXIH0191"

    enable_plots: bool = False
    skip_existing_data: bool = True     # Skip analytic data if residual.pkl exists
    skip_existing_plots: bool = True    # Skip plot generation if plot folder exists
    drop_constant_cm: bool = False
    verbose: bool = True

    # --- DNN configuration ---
    run_dnn_predict: bool = True
    enable_dnn_plots: bool = True
    dnn_model_subpath: str = "dnn/dnn_models/in20__512-512-512-512-64__dr0"
    dnn_model_subpath: list[str] = field(default_factory=lambda: [
        "dnn/dnn_models/in20__512-512-512-512-64__dr0",
        #"dnn/dnn_models/in20__256-256-256-256-32__dr0",
        #"dnn/dnn_models/in20__128-128-128-16__dr0",
        #"dnn/dnn_models/in20__128-128-16__dr0",
        #"dnn/dnn_models/in20__128-128-128-128-16__dr0",
        #"dnn/dnn_models/in20__512-512-512-64__dr0",
        #"dnn/dnn_models/in20__512-512-64__dr0",
        #"dnn/dnn_models/in20__256-256-256-32__dr0",
        #"dnn/dnn_models/in20__256-256-32__dr0",
        #"dnn/dnn_models/in20__64-64-64-64-8__dr0",
        #"dnn/dnn_models/in20__64-64-64-8__dr0",
        #"dnn/dnn_models/in20__64-64-8__dr0"
])

    dnn_weights_name: str = "regression_dnn_best.pth"
    target_modules: list[str] = field(default_factory=lambda: ["ML_F3W_WXIH0191"])

    # Training performance parameters
    enable_performance_plots: bool = False  # Enables/disables performance plots
    enabled_modules_for_plots: list[str] = field(default_factory=lambda: ["ML_F3W_WXIH0191"])  # Modules for which plots will be generated

    # ========== Model configuration ===========
    nodes_per_layer: list[int] = field(default_factory=lambda: [512, 512, 512, 512, 64])
    dropout_rate: float = 0.0
    modeltag: str = ""
    ncmchannels: int = 12
    nch_per_erx: int = 37

    # --- Analytic residuals configuration ---
    enable_analytic_residuals: bool = False
    analytic_residual_modules: list[str] = field(default_factory=lambda: ["ML_F3W_WXIH0192"])


    # --- Analytic + DNN combined plots ---
    enable_combined_dnn_analytic_plots: bool = False
    combined_modules: list[str] = field(default_factory=lambda: ["ML_F3W_WXIH0192"])
    #combined_model_layouts: list[str] = field(default_factory=lambda: ["in20__512-512-512-512-64__dr0"])
    combined_model_layouts: list[str] = field(default_factory=lambda: [
        #"in20__128-128-128-16__dr0",
        #"in20__128-128-16__dr0",
        #"in20__128-128-128-128-16__dr0",
        #"in20__256-256-256-256-32__dr0",
        #"in20__256-256-32__dr0",
        #"in20__256-256-256-32__dr0",
        #"in20__64-64-64-8__dr0",
        #"in20__64-64-8__dr0",
        #"in20__64-64-64-64-8__dr0",
        "in20__512-512-512-512-64__dr0",
        "in20__512-512-512-64__dr0",
        "in20__512-512-64__dr0"
    ])

    # END of Method - A
    enable_coherent_noise_eval: bool = False
    enable_coherent_noise_plots: bool = False
    coherent_noise_modules: list[str] = field(default_factory=lambda: ["ML_F3W_WXIH0191"])


    # --- Method A: comparison charts from results_method_A.txt ---
    enable_methodA_results_comparison: bool = False
    # Two modules will be overlaid as two curves per plot
    methodA_modules_for_comparison: list[str] = field(default_factory=lambda: [
        "ML_F3W_WXIH0191",
        "ML_F3W_WXIH0192",
    ])
    # Where to save comparison charts INSIDE EACH MODULE's folder
    methodA_comparison_subdir: str = "plots/comparison"


    # --- Method B (Residual generation phase) ---
    enable_method_B: bool = False
    method_B_trained_module_dnn1: str = "ML_F3W_WXIH0191"
    method_B_evaluated_module_dnn1: str = "ML_F3W_WXIH0191"
    method_B_model_subpath: str = "in20__512-512-512-512-64__dr0"
    method_B_enable_plots_dnn1: bool = False
    """
    # --- Method B (model inference) ---
    enable_method_B_model_inference: bool = False
    method_B_train_modules: list[str] = field(default_factory=lambda: [
        "ML_F3W_WXIH0190_ML_F3W_WXIH0191_ML_F3W_WXIH0192_ML_F3W_WXIH0193"
    ])
    method_B_eval_modules: list[str] = field(default_factory=lambda: ["ML_F3W_WXIH0191"])
    method_B_models: list[str] = field(default_factory=lambda: ["in20__512-512-512-512-64__dr0"])
    method_B_generate_plots: bool = False
    """

    # ============================================================
    # Method B inference parameters
    # ============================================================
    enable_method_B_prediction: bool = False
    method_B_trained_module: str = "ML_F3W_WXIH0190_ML_F3W_WXIH0191"
    #method_B_eval_modules: List[str] = field(default_factory=lambda: ["ML_F3W_WXIH0191"])
    #method_B_layout_type: Optional[Union[str, List[str]]] = "in20__64-64-64-64-8__dr0"
    method_B_layout_type: Optional[Union[str, List[str]]] = field(default_factory=lambda: [
    #"in20__128-128-128-16__dr0",
    #"in20__128-128-16__dr0",
    #"in20__128-128-128-128-16__dr0",
    #"in20__256-256-256-256-32__dr0",
    #"in20__256-256-32__dr0",
    #"in20__256-256-256-32__dr0",
    #"in20__64-64-64-8__dr0",
    #"in20__64-64-8__dr0",
    #"in20__64-64-64-64-8__dr0",
    "in20__512-512-512-512-64__dr0"
    #"in20__512-512-512-64__dr0",
    #"in20__512-512-64__dr0"
    ])
    method_B_enable_plots: bool = False

    # --- Method B Combined DNN Plots ---
    enable_combined_method_B_plots: bool = False
    method_B_combined_models: list[str] = field(default_factory=lambda: [
        "in20__512-512-512-512-64__dr0"
    ])


    # ===================================================================
    # CONTROL-PREDICT: Independent pretrained model evaluation - DNN ONLY
    # ===================================================================
    enable_control_predict: bool = False

    # Directory where your own pretrained models are stored
    control_model_root: str = "./control_models"


    # Directory containing AREIMERS inputs (always the source of DNN inputs)
    control_input_root: str = "/eos/user/a/areimers/hgcal/dnn_inputs"

    # Output directory where predictions and plots will be saved
    control_output_root: str = "/eos/user/g/gmihriye/hgcal/CM/control_models/outputs"

    control_model_layouts: list[str] = field(default_factory=lambda: [
        "in20__128-128-128-16__dr0",
        "in20__128-128-16__dr0",
        "in20__128-128-128-128-16__dr0",
        "in20__256-256-256-256-32__dr0",
        "in20__256-256-32__dr0",
        "in20__256-256-256-32__dr0",
        "in20__64-64-64-8__dr0",
        "in20__64-64-8__dr0",
        "in20__64-64-64-64-8__dr0",
        "in20__512-512-512-512-64__dr0",
        "in20__512-512-512-64__dr0",
        "in20__512-512-64__dr0"
    ])

    # List of module names that were used during training of these models
    control_trained_modules: list[str] = field(default_factory=lambda: [
        "ML_F3W_WXIH0191"
    ])

    # List of module names for which predictions will be generated
    control_evaluated_modules: list[str] = field(default_factory=lambda: [
        "ML_F3W_WXIH0191",
    ])

    # Common DNN hyperparameters for creating the model skeleton
    control_dropout_rate: float = 0.0
    control_ncmchannels: int = 12
    control_nch_per_erx: int = 37

    # Enable/disable control-predict plotting
    enable_control_plots: bool = False


    # ============================================================
    # CONTROL-PREDICT: Fine-tuned pretrained model evaluation
    # ============================================================

    enable_control_predict_ftuning: bool = True
    enable_control_plots_ftuning: bool = False

    # --- Fine-tuning pretrained models ---
    control_model_root_ftuning: str = (
        "/eos/user/g/gmihriye/hgcal/CM/dnn/fine_tuning/"
        "ML_F3W_WXIH0190_ML_F3W_WXIH0191_ML_F3W_WXIH0192_ML_F3W_WXIH0193/"
        "ML_F3W_WXIH0192/dnn_models"
    )

    # --- Where prediction PKLs will be written ---
    control_output_root_ftuning: str = (
        "/eos/user/g/gmihriye/CM/compare_methods/CMCorrection/method_C-finetuning/"
        "trained_ML_F3W_WXIH0190_ML_F3W_WXIH0191_ML_F3W_WXIH0192_ML_F3W_WXIH0193/"
        "evaluated_ML_F3W_WXIH0192/dnn/dnn_outputs"
    )

    # --- Where plots will be written ---
    control_plots_root_ftuning: str = (
        "/eos/user/g/gmihriye/CM/compare_methods/CMCorrection/plots/performance/"
        "finetuning/pretrained_ML_F3W_WXIH0190_ML_F3W_WXIH0191_ML_F3W_WXIH0192_ML_F3W_WXIH0193"
    )

    # --- Input source ---
    control_input_root_ftuning: str = "/eos/user/a/areimers/hgcal/dnn_inputs"


    # --- Model layouts ---
    control_model_layouts_ftuning: list[str] = field(default_factory=lambda: [
        #"in20__128-128-128-16__dr0",
        #"in20__128-128-16__dr0",
        "in20__128-128-128-128-16__dr0",
        "in20__256-256-256-256-32__dr0",
        "in20__256-256-32__dr0",
        "in20__256-256-256-32__dr0",
        "in20__64-64-64-8__dr0",
        "in20__64-64-8__dr0",
        "in20__64-64-64-64-8__dr0",
        "in20__512-512-512-512-64__dr0",
        "in20__512-512-512-64__dr0",
        "in20__512-512-64__dr0"
    ])

    # --- Trained modules (only used for naming) ---
    control_trained_modules_ftuning: list[str] = field(default_factory=lambda: [
        "ML_F3W_WXIH0192"
    ])

    # --- Evaluated modules ---
    control_evaluated_modules_ftuning: list[str] = field(default_factory=lambda: [
        "ML_F3W_WXIH0192"
    ])

    # --- DNN hyperparameters ---
    control_dropout_rate: float = 0.0
    control_ncmchannels: int = 12
    control_nch_per_erx: int = 37


# ============================================================
# Helper Functions
# ============================================================

def get_valid_modules(base_dir: str) -> list[str]:
    """Scan the dnn_inputs directory and return valid module folders."""
    valid_modules = []
    for module in sorted(os.listdir(base_dir)):
        module_path = os.path.join(base_dir, module)
        if not os.path.isdir(module_path):
            continue
        if len(os.listdir(module_path)) == 0:
            print(f"[WARNING] Skipping empty module folder: {module}")
            continue
        valid_modules.append(module)
    return valid_modules

def _flatten_str_list(x) -> list[str]:
    """
    Accepts str or nested iterables (list/tuple/set of str),
    returns a flat list[str]. None'ları atar, diğer tipleri str'e çevirir.
    """
    if isinstance(x, str):
        return [x]
    out = []
    def rec(v):
        if v is None:
            return
        if isinstance(v, (list, tuple, set)):
            for t in v:
                rec(t)
        else:
            out.append(str(v))
    rec(x)
    return out


def check_colnames_consistency(module_dir: str, reference_module_dir: str) -> str:
    """Check if colnames.json matches input dimension; fallback to reference if mismatched."""
    colnames_path = os.path.join(module_dir, "colnames.json")
    inputs_path = os.path.join(module_dir, "inputs_train.npy")

    if not (os.path.exists(colnames_path) and os.path.exists(inputs_path)):
        print(f"[WARNING] Missing files in {module_dir}, skipping consistency check.")
        return colnames_path

    try:
        with open(colnames_path, "r") as f:
            colnames = json.load(f)
        inputs = np.load(inputs_path, mmap_mode="r")
        if len(colnames) != inputs.shape[1]:
            print(f"[WARNING] Feature count mismatch in {module_dir}: "
                  f"{len(colnames)} names vs {inputs.shape[1]} columns in inputs.")
            ref_colnames_path = os.path.join(reference_module_dir, "colnames.json")
            print(f"[INFO] Replacing colnames.json of {module_dir} "
                  f"with reference from {os.path.basename(reference_module_dir)}.")
            return ref_colnames_path
    except Exception as e:
        print(f"[ERROR] Failed to check colnames consistency for {module_dir}: {e}")

    return colnames_path


def _parse_model_layout(model_layout: str) -> dict:
    """
    Parse strings like 'in20__512-512-512-512-64__dr0' or '512-512-64__dr0' → layer sizes etc.
    Returns:
      {
        "layers": [512,512,512,512,64],
        "layer_count": 5,
        "family_base": 512,   # max(layers) as family
        "family_tail": 64,    # last layer
        "nodes_signature": "512-512-512-512-64"
      }
    """
    name = model_layout
    if "__" in name:
        try:
            middle = name.split("__")[1]          # e.g. 512-512-512-512-64
        except Exception:
            middle = name
    else:
        # strip suffix __dr0 if present
        middle = name.split("__")[0]

    # strip trailing __drX if still present
    middle = middle.split("__")[0]
    middle = middle.replace("_", "-")  # be tolerant

    # keep only numbers and dashes
    parts = []
    for token in middle.split("-"):
        token = "".join([c for c in token if c.isdigit()])
        if token.isdigit():
            parts.append(int(token))
    layers = parts[:] if parts else []

    nodes_sig = "-".join(str(x) for x in layers) if layers else middle
    layer_count = len(layers)
    family_base = max(layers) if layers else None
    family_tail = layers[-1] if layers else None

    return {
        "layers": layers,
        "layer_count": layer_count,
        "family_base": family_base,
        "family_tail": family_tail,
        "nodes_signature": nodes_sig,
    }


def _nice_xtick_from_nodes(nodes_sig: str) -> str:
    """
    Compact x label like '512-512-512-512-64' → '512×4→64'
    '256-256-256-32'     → '256×3→32'
    Falls back to original if pattern not obvious.
    """
    try:
        nums = [int(x) for x in nodes_sig.split("-")]
        if len(nums) >= 2:
            base = nums[0]
            tail = nums[-1]
            run = 0
            for v in nums:
                if v == base:
                    run += 1
                else:
                    break
            if run >= 2:
                return f"{base}×{run}→{tail}"
    except Exception:
        pass
    return nodes_sig


"""

def run_dnn_prediction_for_module(cfg, module):
    import torch
    from models import DNNFlex
    from evaluate_performance import DNNInferencer, pivot_flat_preds_to_event_channel

    print(f"\n[DNN] Running DNN prediction for module: {module}")


    model_dir = os.path.join(cfg.base_output_dir, module, cfg.dnn_model_subpath)
    weights_path = os.path.join(model_dir, cfg.dnn_weights_name)

    # === Create unique output folder per model architecture ===
    # Example:
    # ./method_A/ML_F3W_WXIH0192/dnn/dnn_outputs/in20__512-512-512-512-64__dr0/
    model_folder_name = (
        cfg.dnn_model_subpath
        .replace("dnn/dnn_models/", "")
        .replace("/", "_")
    )
    dnn_output_dir = os.path.join(
        cfg.base_output_dir,
        module,
        "dnn",
        "dnn_outputs",
        model_folder_name
    )
    os.makedirs(dnn_output_dir, exist_ok=True)


    if not os.path.exists(weights_path):
        print(f"[SKIP] No DNN weights found for {module}: {weights_path}")
        return

    # === Read DNN inputs generated by our own pipeline (not EOS) ===
    dnn_input_dir = os.path.join(cfg.base_output_dir, module, "dnn", "dnn_inputs")
    # Create separate directory for modified prediction inputs
    dnn_predict_dir = os.path.join(cfg.base_output_dir, module, "dnn", "dnn_predict")
    os.makedirs(dnn_predict_dir, exist_ok=True)

    if not os.path.exists(dnn_input_dir):
        raise FileNotFoundError(f"[ERROR] DNN input directory not found: {dnn_input_dir}")


    # ============================================================
    # --- Load DNN input and target data (analytic residuals) ---
    # ============================================================

    dnn_input_dir = os.path.join(cfg.base_output_dir, module, "dnn", "dnn_inputs")

    # --- Load original DNN inputs (20 features) ---
    inputs_train = np.load(os.path.join(dnn_input_dir, "inputs_train.npy"))
    inputs_val   = np.load(os.path.join(dnn_input_dir, "inputs_val.npy"))
    inputs_combined = np.concatenate([inputs_train, inputs_val], axis=0)
    print(f"[INFO] Loaded DNN inputs from {dnn_input_dir}")
    print(f"[INFO] Input shape: {inputs_combined.shape}")

    # --- Load analytic residuals (targets) ---
    targets_train = np.load(os.path.join(dnn_input_dir, "targets_train.npy"))
    targets_val   = np.load(os.path.join(dnn_input_dir, "targets_val.npy"))
    targets_combined = np.concatenate([targets_train, targets_val], axis=0)
    print(f"[INFO] Loaded analytic residuals (targets). Shape: {targets_combined.shape}")

    # --- Column names and index check ---
    colnames_path = os.path.join(dnn_input_dir, "colnames.json")
    with open(colnames_path, "r") as f:
        colnames = json.load(f)

    inputs_for_predict = inputs_combined

    print(f"[INFO] Using DNN inputs from {dnn_input_dir}")
    print(f"[INFO] Combined input shape: {inputs_combined.shape}, target shape: {targets_combined.shape}")


    eval_cfg = ep.EvalConfig(
        modulenames_used_for_training=[module],
        modulename_for_evaluation=module,
        nodes_per_layer=cfg.nodes_per_layer,
        dropout_rate=cfg.dropout_rate,
        modeltag=cfg.modeltag,
        inputfoldertag="",
        ncmchannels=cfg.ncmchannels,
        nch_per_erx=cfg.nch_per_erx,
    )

    io = ep.DataIO(eval_cfg)
    io.load_all()
    split = io.get_split("combined")

    input_dim = inputs_combined.shape[1]
    #model = DNNFlex(input_dim, cfg.nodes_per_layer, cfg.dropout_rate)
    model = DNNFlex(input_dim, layer_sizes, cfg.dropout_rate)
    inferencer = ep.DNNInferencer(model=model, weights_path=weights_path, dtype=np.float32, batch_size=16384)


    # --- Load column names and locate 'chadc' column ---
    colnames_path = os.path.join(dnn_input_dir, "colnames.json")
    with open(colnames_path, "r") as f:
        colnames = json.load(f)
    idx_chadc = colnames.index("chadc")

    inputs_for_predict = inputs_combined
    print(f"[INFO] Using original DNN inputs (no modification). Shape: {inputs_for_predict.shape}")


    # --- Load from dnn_predict to ensure correct source ---
    #inputs_for_predict = np.load(os.path.join(dnn_predict_dir, "inputs_modified.npy"))
    preds_flat = inferencer(inputs_for_predict.astype(np.float32))

    preds_df = pivot_flat_preds_to_event_channel(preds_flat, split.eventid_flat, split.channels_flat, split.measurements_df)

    np.save(os.path.join(dnn_output_dir, f"predictions_{module}_combined.npy"), preds_flat)
    preds_df.to_pickle(os.path.join(dnn_output_dir, f"predictions_{module}_combined.pkl"))

    preds_df_with_cm = ep.add_cms_to_measurements_df(preds_df, split.cm_df, drop_constant_cm=False)
    preds_df_with_cm.to_pickle(os.path.join(dnn_output_dir, f"predictions_{module}_withCM_combined.pkl"))
    print(f"[OK] DNN predictions (with CM) saved → {dnn_output_dir}")

    meas_true_path = os.path.join(cfg.base_output_dir, module, "analytic", "meas_true.pkl")
    if not os.path.exists(meas_true_path):
        print(f"[WARN] meas_true.pkl missing for {module}, skipping residual computation.")
        return

    meas_true = pd.read_pickle(meas_true_path)
    meas_true = meas_true.loc[preds_df.index]

    residual_dnn = meas_true - preds_df
    residual_dnn_with_cm = ep.add_cms_to_measurements_df(residual_dnn, split.cm_df, drop_constant_cm=False)

    residual_dnn.to_pickle(os.path.join(dnn_output_dir, f"residual_dnn_{module}.pkl"))
    residual_dnn_with_cm.to_pickle(os.path.join(dnn_output_dir, f"residual_dnn_{module}_withCM.pkl"))
    print(f"[OK] DNN residuals (with CM) saved → {dnn_output_dir}")

    # --- Compute final residual = (true - analytic_pred) - dnn_pred ---
    analytic_pred_path = os.path.join(cfg.base_output_dir, module, "analytic", "analytic_pred.pkl")
    if os.path.exists(analytic_pred_path):
        analytic_pred = pd.read_pickle(analytic_pred_path)
        analytic_pred = analytic_pred.loc[preds_df.index]

        final_residual = (meas_true - analytic_pred) - preds_df
        final_residual_with_cm = ep.add_cms_to_measurements_df(final_residual, split.cm_df, drop_constant_cm=False)

        final_residual.to_pickle(os.path.join(dnn_output_dir, f"final_residual_{module}.pkl"))
        final_residual_with_cm.to_pickle(os.path.join(dnn_output_dir, f"final_residual_{module}_withCM.pkl"))
        print(f"[OK] Final residual (with CM) saved → {dnn_output_dir}")
    else:
        print(f"[WARN] analytic_pred.pkl not found, skipping final residual computation.")

"""

def _load_results_methodA(base_output_dir: str, module: str) -> pd.DataFrame:
    """
    Load ./method_A/<module>/results_method_A.txt → DataFrame
    Ensures the parsed columns exist and adds parsed model metadata.
    """
    path = os.path.join(base_output_dir, module, "results_method_A.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"[MISS] {path}")

    df = pd.read_csv(path)
    # expected columns: module,model_layout,dropout,nch_per_erx,rms_before,rms_after,frac_improvement,coh_ratio_mean
    if "model_layout" not in df.columns or "coh_ratio_mean" not in df.columns:
        raise ValueError(f"[ERR] Unexpected columns in {path}: {list(df.columns)}")

    parsed = df["model_layout"].apply(_parse_model_layout)
    df["nodes_signature"] = parsed.apply(lambda d: d["nodes_signature"])
    df["layer_count"] = parsed.apply(lambda d: d["layer_count"])
    df["family_base"] = parsed.apply(lambda d: d["family_base"])
    df["family_tail"] = parsed.apply(lambda d: d["family_tail"])
    df["xtick"] = df["nodes_signature"].apply(_nice_xtick_from_nodes)
    return df


def _ensure_save_in_both_modules(fig, cfg, modules: list[str], filename: str):
    """
    Save the same figure into each module's comparison folder.
    """
    for m in modules:
        out_dir = os.path.join(cfg.base_output_dir, m, cfg.methodA_comparison_subdir)
        os.makedirs(out_dir, exist_ok=True)
        figpath = os.path.join(out_dir, filename)
        fig.savefig(figpath, dpi=180, bbox_inches="tight")
        print(f"[SAVE] {figpath}")


def _style_series(ax, x, y, label):
    """
    Visual styling: markers, line, value labels on each point.
    """
    ln, = ax.plot(x, y, marker="o", linewidth=2.0, label=label)
    # annotate values on the line (on-point labels)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.3f}", (xi, yi),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)


def generate_methodA_comparison_plots(cfg):
    """
    Build 4 charts from results_method_A.txt combining two modules on each plot.

    Chart set:
      (A1) Fixed layer-count = 5  → compare families by nodes_signature on X
      (A2) Fixed layer-count = 4  → compare families by nodes_signature on X
      (B1) Fixed node family = 512 → vary layer-count (2/3/4/5), X=nodes_signature (compact)
      (B2) Fixed node family = 128 → vary layer-count, X=nodes_signature

    Y axis: Coherent noise ratio (corr/true)  [= coh_ratio_mean in file]
    """
    if not getattr(cfg, "enable_methodA_results_comparison", False):
        print("[SKIP] Method A results comparison disabled.")
        return

    modules = cfg.methodA_modules_for_comparison
    if len(modules) < 2:
        print("[WARN] Need at least two modules to overlay; proceeding with whatever is available.")

    # Load per-module dataframes
    dfs = {}
    for m in modules:
        try:
            dfs[m] = _load_results_methodA(cfg.base_output_dir, m)
            print(f"[OK] Loaded results for {m}: {len(dfs[m])} rows")
        except Exception as e:
            print(f"[SKIP] {m}: {e}")

    if not dfs:
        print("[SKIP] No results loaded.")
        return

    # --- Common helper to draw overlay (updated: PDF + styled titles) ---
    def draw_overlay(dfA, dfB, title, filename, x_order_cols, title_type):
        import matplotlib.pyplot as plt

        # Global styling
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        })

        fig, ax = plt.subplots(figsize=(6, 4.2))  # more compact layout
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
        ax.set_ylabel("Coherent noise ratio (corr/true)")
        ax.set_xlabel("Model (layers → tail)")

        # Dynamic title selection
        if title_type == "node":
            ax.set_title(f"Coherent noise ratio – Node width variation {title}", pad=10)
        elif title_type == "layer":
            ax.set_title(f"Coherent noise ratio – Layer count variation {title}", pad=10)
        else:
            ax.set_title(title, pad=10)

        # keep consistent x order
        xlabels = x_order_cols
        xpos = np.arange(len(xlabels))

        def y_for(df, order):
            m = df.set_index("nodes_signature")["coh_ratio_mean"].to_dict()
            return [m.get(k, np.nan) for k in order]

        # Series A (module 1)
        keyA = modules[0] if modules else "ModuleA"
        if dfA is not None and len(dfA) > 0:
            yA = y_for(dfA, xlabels)
            ln, = ax.plot(xpos, yA, marker="D", markersize=6.5,
                          linewidth=2.2, label=keyA)
            for xi, yi in zip(xpos, yA):
                ax.annotate(f"{yi:.3f}", (xi, yi),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=9)

        # Series B (module 2)
        keyB = (modules[1] if len(modules) > 1 else None)
        if keyB and keyB in dfs:
            dfB2 = dfs[keyB]
            mask = dfB2["nodes_signature"].isin(xlabels)
            dfBf = dfB2[mask]
            yB = y_for(dfBf, xlabels)
            ln, = ax.plot(xpos, yB, marker="D", markersize=6.5,
                          linewidth=2.2, label=keyB)
            for xi, yi in zip(xpos, yB):
                ax.annotate(f"{yi:.3f}", (xi, yi),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=9)

        # x ticks
        xt = [_nice_xtick_from_nodes(x) for x in xlabels]
        ax.set_xticks(xpos)
        ax.set_xticklabels(xt, rotation=0)
        ax.legend(frameon=False)

        fig.tight_layout()

        # --- Save as PDF (for both modules) ---
        for m in modules:
            out_dir = os.path.join(cfg.base_output_dir, m, cfg.methodA_comparison_subdir)
            os.makedirs(out_dir, exist_ok=True)
            pdf_name = filename.replace(".png", ".pdf")
            figpath = os.path.join(out_dir, pdf_name)
            fig.savefig(figpath, dpi=300, bbox_inches="tight", format="pdf")
            print(f"[SAVE] {figpath}")

        plt.close(fig)


    # ====== (A1) & (A2): Fixed layer-count ======
    for fixed_layers, fname in [
        (5, "A1_node_variation_5layers.pdf"),
        (4, "A2_node_variation_4layers.pdf"),
    ]:
        # Build x-order from union of modules, preserving a nice sort by family desc then tail
        union = []
        for m, df in dfs.items():
            union.append(df[df["layer_count"] == fixed_layers][["nodes_signature","family_base","family_tail"]])
        if not union:
            continue
        uni = pd.concat(union, ignore_index=True).drop_duplicates("nodes_signature")
        if len(uni) == 0:
            print(f"[INFO] No models with layer_count={fixed_layers}")
            continue
        uni = uni.sort_values(["family_base", "family_tail"], ascending=[False, True])
        x_order = uni["nodes_signature"].tolist()

        dfA = dfs[modules[0]][dfs[modules[0]]["layer_count"] == fixed_layers] if modules[0] in dfs else None
        dfB = dfs[modules[1]][dfs[modules[1]]["layer_count"] == fixed_layers] if len(modules) > 1 and modules[1] in dfs else None

        ttl = f"Method A | Fixed layer count = {fixed_layers} | Y: Coherent noise ratio (corr/true)"
        draw_overlay(dfA, dfB, ttl, fname, x_order, title_type="node")

    # ====== (B1) & (B2): Fixed node family (512 and 128) ======
    for fam, fname in [
        (512, "B1_layer_variation_512nodes.pdf"),
        (128, "B2_layer_variation_128nodes.pdf"),
    ]:
        union = []
        for m, df in dfs.items():
            union.append(df[df["family_base"] == fam][["nodes_signature","layer_count","family_tail"]])
        if not union:
            continue
        uni = pd.concat(union, ignore_index=True).drop_duplicates("nodes_signature")
        if len(uni) == 0:
            print(f"[INFO] No models with family={fam}")
            continue
        # order: increasing layer_count, then tail
        uni = uni.sort_values(["layer_count", "family_tail"], ascending=[True, True])
        x_order = uni["nodes_signature"].tolist()

        dfA = dfs[modules[0]][dfs[modules[0]]["family_base"] == fam] if modules[0] in dfs else None
        dfB = dfs[modules[1]][dfs[modules[1]]["family_base"] == fam] if len(modules) > 1 and modules[1] in dfs else None

        ttl = f"Method A | Family {fam} (vary layers) | Y: Coherent noise ratio (corr/true)"
        draw_overlay(dfA, dfB, ttl, fname, x_order, title_type="layer")


def run_dnn_prediction_for_module(cfg, module):
    """
    Run DNN prediction for one or more model architectures for a given module.
    Each model path in cfg.dnn_model_subpath will be processed sequentially.
    Outputs (predictions, residuals, final residuals) are stored under:
      <base_output_dir>/<module>/dnn/dnn_outputs/<model_subpath>/
    """
    import torch
    import numpy as np
    import pandas as pd
    import os, json
    from models import DNNFlex
    from evaluate_performance import DNNInferencer, pivot_flat_preds_to_event_channel
    import evaluate_performance as ep

    # --- Support for multiple model architectures ---
    model_subpaths = cfg.dnn_model_subpath
    if isinstance(model_subpaths, str):
        model_subpaths = [model_subpaths]

    for model_subpath in model_subpaths:
        print(f"\n[DNN] Running DNN prediction for module: {module} | model: {model_subpath}")

        # --- Define paths ---
        model_dir = os.path.join(cfg.base_output_dir, module, model_subpath)
        weights_path = os.path.join(model_dir, cfg.dnn_weights_name)

        # --- Output folder ---
        model_folder_name = (
            model_subpath
            .replace("dnn/dnn_models/", "")
            .replace("/", "_")
        )
        dnn_output_dir = os.path.join(
            cfg.base_output_dir,
            module,
            "dnn",
            "dnn_outputs",
            model_folder_name
        )
        os.makedirs(dnn_output_dir, exist_ok=True)

        if not os.path.exists(weights_path):
            print(f"[SKIP] No DNN weights found for {module}: {weights_path}")
            continue

        # --- Load inputs and targets ---
        dnn_input_dir = os.path.join(cfg.base_output_dir, module, "dnn", "dnn_inputs")
        if not os.path.exists(dnn_input_dir):
            raise FileNotFoundError(f"[ERROR] DNN input directory not found: {dnn_input_dir}")

        inputs_train = np.load(os.path.join(dnn_input_dir, "inputs_train.npy"))
        inputs_val = np.load(os.path.join(dnn_input_dir, "inputs_val.npy"))
        inputs_combined = np.concatenate([inputs_train, inputs_val], axis=0)

        targets_train = np.load(os.path.join(dnn_input_dir, "targets_train.npy"))
        targets_val = np.load(os.path.join(dnn_input_dir, "targets_val.npy"))
        targets_combined = np.concatenate([targets_train, targets_val], axis=0)

        print(f"[INFO] Loaded DNN inputs from {dnn_input_dir}")
        print(f"[INFO] Input shape: {inputs_combined.shape}, Target shape: {targets_combined.shape}")

        # --- Build evaluation config ---
        eval_cfg = ep.EvalConfig(
            modulenames_used_for_training=[module],
            modulename_for_evaluation=module,
            nodes_per_layer=cfg.nodes_per_layer,
            dropout_rate=cfg.dropout_rate,
            modeltag=cfg.modeltag,
            inputfoldertag="",
            ncmchannels=cfg.ncmchannels,
            nch_per_erx=cfg.nch_per_erx,
        )

        io = ep.DataIO(eval_cfg)
        io.load_all()
        split = io.get_split("combined")

        # --- Model definition ---
        input_dim = inputs_combined.shape[1]
        #model = DNNFlex(input_dim, cfg.nodes_per_layer, cfg.dropout_rate)
        # --- Parse layer sizes from model_subpath ---
        try:
            arch_str = model_subpath.split("__")[1]  # örn. '256-256-256-256-32'
            layer_sizes = [int(x) for x in arch_str.split("-") if x.isdigit()]
            print(f"[INFO] Parsed architecture from model name: {layer_sizes}")
        except Exception:
            layer_sizes = cfg.nodes_per_layer
            print(f"[WARN] Could not parse architecture from name, using default: {layer_sizes}")

        # --- Create model with correct layer sizes ---
        model = DNNFlex(input_dim, layer_sizes, cfg.dropout_rate)

        #inferencer = ep.DNNInferencer(model=model, weights_path=weights_path, dtype=np.float32, batch_size=16384)

        # ============================================================
        # RAW INFERENCE (same method as CONTROL pipeline)
        # ============================================================

        n_samples = inputs_combined.shape[0]

        # --- FLOP calculation using the existing function ---
        flops_per_sample = estimate_dense_flops_per_sample(
            input_dim=input_dim,
            hidden_nodes=layer_sizes,
            output_dim=1
        )
        total_flops = flops_per_sample * n_samples

        # Load model weights
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        # Use the same batch size as CONTROL pipeline
        batch_size = 20000

        # Start performance timers
        t0_wall = time.perf_counter()
        t0_cpu  = time.process_time()

        # Convert inputs to tensor
        X_tensor = torch.tensor(inputs_combined, dtype=torch.float32)
        pred_list = []

        # Run prediction in batches
        with torch.no_grad():
            for i in range(0, X_tensor.shape[0], batch_size):
                batch = X_tensor[i:i + batch_size]
                pred_batch = model(batch).cpu().numpy()
                pred_list.append(pred_batch)

        # Concatenate all predictions
        preds_flat = np.concatenate(pred_list, axis=0).flatten()

        # Stop performance timers
        t1_wall = time.perf_counter()
        t1_cpu  = time.process_time()

        wall_time = t1_wall - t0_wall
        cpu_time  = t1_cpu - t0_cpu

        throughput = n_samples / wall_time
        gflops = total_flops / 1e9
        gflops_per_sec = gflops / wall_time

        # --- Run prediction ---
        #preds_flat = inferencer(inputs_combined.astype(np.float32))


        # ============================================================
        # SAVE PERFORMANCE.LOG
        # ============================================================
        perf_path = os.path.join(dnn_output_dir, "performance.log")
        with open(perf_path, "w") as f:
            f.write(f"module: {module}\n")
            f.write(f"model: {model_subpath}\n")
            f.write(f"samples: {n_samples}\n")
            f.write(f"wall_time: {wall_time:.6f}\n")
            f.write(f"cpu_time:  {cpu_time:.6f}\n")
            f.write(f"throughput_samples_per_sec: {throughput:.2f}\n")
            f.write(f"flops_per_sample: {flops_per_sample}\n")
            f.write(f"total_flops: {total_flops}\n")
            f.write(f"gflops_per_sec: {gflops_per_sec:.4f}\n")

        print(f"[PERF] Performance log saved → {perf_path}")

        preds_df = pivot_flat_preds_to_event_channel(
            preds_flat, split.eventid_flat, split.channels_flat, split.measurements_df
        )

        # --- Save predictions ---
        np.save(os.path.join(dnn_output_dir, f"predictions_{module}_combined.npy"), preds_flat)
        preds_df.to_pickle(os.path.join(dnn_output_dir, f"predictions_{module}_combined.pkl"))

        preds_df_with_cm = ep.add_cms_to_measurements_df(preds_df, split.cm_df, drop_constant_cm=False)
        preds_df_with_cm.to_pickle(os.path.join(dnn_output_dir, f"predictions_{module}_withCM_combined.pkl"))
        print(f"[OK] DNN predictions (with CM) saved → {dnn_output_dir}")

        # --- Load true measurements ---
        meas_true_path = os.path.join(cfg.base_output_dir, module, "analytic", "meas_true.pkl")
        if not os.path.exists(meas_true_path):
            print(f"[WARN] meas_true.pkl missing for {module}, skipping residual computation.")
            continue

        meas_true = pd.read_pickle(meas_true_path)
        meas_true = meas_true.loc[preds_df.index]

        # --- Compute DNN residuals ---
        residual_dnn = meas_true - preds_df
        residual_dnn_with_cm = ep.add_cms_to_measurements_df(residual_dnn, split.cm_df, drop_constant_cm=False)
        residual_dnn.to_pickle(os.path.join(dnn_output_dir, f"residual_dnn_{module}.pkl"))
        residual_dnn_with_cm.to_pickle(os.path.join(dnn_output_dir, f"residual_dnn_{module}_withCM.pkl"))
        print(f"[OK] DNN residuals saved → {dnn_output_dir}")

        # --- Compute final residual (true - analytic_pred - dnn_pred) ---
        analytic_pred_path = os.path.join(cfg.base_output_dir, module, "analytic", "analytic_pred.pkl")
        if os.path.exists(analytic_pred_path):
            analytic_pred = pd.read_pickle(analytic_pred_path)
            analytic_pred = analytic_pred.loc[preds_df.index]

            final_residual = (meas_true - analytic_pred) - preds_df
            final_residual_with_cm = ep.add_cms_to_measurements_df(final_residual, split.cm_df, drop_constant_cm=False)

            final_residual.to_pickle(os.path.join(dnn_output_dir, f"final_residual_{module}.pkl"))
            final_residual_with_cm.to_pickle(os.path.join(dnn_output_dir, f"final_residual_{module}_withCM.pkl"))
            print(f"[OK] Final residuals saved → {dnn_output_dir}")
        else:
            print(f"[WARN] analytic_pred.pkl not found, skipping final residual computation.")



# ============================================================
# Main Pipeline
# ============================================================

def compare_methods(cfg: CompareConfig):
    """Run the analytic → residual → DNN preparation pipeline."""
    print("\n[INIT] Starting compare_methods pipeline...\n")

    valid_modules = get_valid_modules(cfg.base_input_dir)
    if len(valid_modules) == 0:
        raise RuntimeError("No valid module folders found.")
    print(f"[SCAN] Found {len(valid_modules)} modules: {valid_modules}")

    # === Performance plot control ===
    if cfg.enable_performance_plots:
        print("\n[INFO] Generating DNN performance plots (loss curves)...")
        for module in cfg.enabled_modules_for_plots:
            if module not in valid_modules:
                print(f"[WARN] Module {module} not found in valid modules, skipping.")
                continue
            plot_loss_for_module(cfg, module)
        print("[OK] Performance plot generation completed.\n")
    else:
        print("[INFO] Performance plot generation disabled by config.")
    # ============================================

    reference_module_dir = os.path.join(cfg.base_input_dir, cfg.reference_module)
    if not os.path.exists(reference_module_dir):
        raise RuntimeError(f"Reference module {cfg.reference_module} not found!")

    # --- Create one AnalyticInferencer globally (to keep cache consistent)
    analytic = ep.AnalyticInferencer(drop_constant_cm=cfg.drop_constant_cm)

    for module in valid_modules:
        if cfg.verbose:
            print(f"\n[RUN] Processing module: {module}")

        module_dir = os.path.join(cfg.base_input_dir, module)
        analytic_output_dir = os.path.join(cfg.base_output_dir, module, "analytic")
        dnn_output_dir = os.path.join(cfg.base_output_dir, module, "dnn", "dnn_inputs")
        os.makedirs(analytic_output_dir, exist_ok=True)
        os.makedirs(dnn_output_dir, exist_ok=True)

        # --- Step 3: Skip existing analytic data
        residual_file = os.path.join(analytic_output_dir, "residual.pkl")
        data_exists = os.path.exists(residual_file)

        if cfg.skip_existing_data and data_exists:
            print(f"[SKIP] Analytic data already exist for {module}, skipping analytic computation.")
            # --- Still run DNN prediction if enabled ---
            #if cfg.run_dnn_predict and (cfg.target_modules is None or module in cfg.target_modules):
            #    print(f"[INFO] Running DNN prediction for existing analytic data in {module}...")
            #    run_dnn_prediction_for_module(cfg, module)
            #continue
        else:
            # --- Step 4: Colnames consistency check
            fixed_colnames_path = check_colnames_consistency(module_dir, reference_module_dir)

            # --- Step 5: Prepare analytic configuration
            eval_cfg = ep.EvalConfig(
                modulenames_used_for_training=[module],
                modulename_for_evaluation=module,
                nodes_per_layer=cfg.nodes_per_layer,
                dropout_rate=cfg.dropout_rate,
                modeltag=cfg.modeltag,
                inputfoldertag="",
                ncmchannels=cfg.ncmchannels,
                nch_per_erx=cfg.nch_per_erx,
            )

            # --- Step 6: Run analytic computation
            io = ep.DataIO(eval_cfg)
            io.load_all()
            split = io.get_split("combined")

            # --- Detect CM drop candidates and add epsilon (prevent drop) ---
            cm_df = split.cm_df.copy()
            cm_var = cm_df.var(axis=0)
            zero_var_cols = cm_var[cm_var == 0].index.tolist()

            if len(zero_var_cols) > 0:
                eps = 1e-8
                print(f"[INFO] {len(zero_var_cols)} CM channels have zero variance → adding epsilon={eps:.1e}")
                for col in zero_var_cols:
                    cm_df[col] = cm_df[col] + np.random.normal(0, eps, size=len(cm_df))
            else:
                print("[INFO] No CM channels with zero variance detected; proceeding normally.")

            # Update split.cm_df with the epsilon-regularized CM data
            split.cm_df = cm_df

            # Force analytic to keep all CM channels
            analytic.drop_constant_cm = False


            analytic.fit(split)
            analytic_pred = analytic.predict(split_predict=split, split_correction=split)
            meas_true = split.measurements_df


            assert np.array_equal(meas_true.index, analytic_pred.index)


            residual = meas_true - analytic_pred

            # --- Create DNN-specific residual copy (with epsilon for dropped CMs) ---
            residual_tar = residual.copy()

            # Apply epsilon to dropped CM columns if CM drop was used
            if hasattr(analytic, "keep_mask") and analytic.keep_mask is not None:
                dropped_cols = residual.columns[~analytic.keep_mask]
                if len(dropped_cols) > 0:
                    eps = 1e-8
                    print(f"[INFO] Adding epsilon to {len(dropped_cols)} dropped CM columns for residual_tar...")
                    noise = np.random.normal(0, eps, size=(len(residual), len(dropped_cols)))
                    residual_tar[dropped_cols] = residual_tar[dropped_cols].add(noise, fill_value=0)

            # Flatten residual_tar for DNN targets
            if isinstance(residual_tar.index, pd.MultiIndex) or residual_tar.ndim == 2:
                residual_tar = residual_tar.stack().to_frame(name="adc")
            elif isinstance(residual_tar, pd.Series):
                residual_tar = residual_tar.to_frame(name="adc")

            # Reset index to avoid MultiIndex concat issues

            # Save DNN-specific residual version
            residual_tar_path = os.path.join(analytic_output_dir, "residual_tar.pkl")
            residual_tar.to_pickle(residual_tar_path)
            print(f"[OK] DNN residual_tar saved → {residual_tar_path}")



            # --- Step 7: Save analytic outputs
            meas_true.to_pickle(os.path.join(analytic_output_dir, "meas_true.pkl"))
            analytic_pred.to_pickle(os.path.join(analytic_output_dir, "analytic_pred.pkl"))
            residual.to_pickle(os.path.join(analytic_output_dir, "residual.pkl"))
            print(f"[OK] Analytic results saved → {analytic_output_dir}")

        # --- Skip DNN preparation if analytic step was skipped ---
        if cfg.skip_existing_data and data_exists:
            print(f"[SKIP] Skipping DNN input/target creation for {module} (analytic already exists).")
            continue



        # --- Step 8: Create DNN-ready data (inputs=20 features from split.inputs_df, targets=residual) ---

        print(f"[INFO] Creating DNN inputs & targets aligned to residual (CM-dropped alignment)...")

        # 1) Gerekirse residual'ı (n_events, n_channels) -> (event_id, channel) MultiIndex'e flatten et
        if isinstance(residual_tar.index, pd.RangeIndex) or residual_tar.ndim == 2:
            # Kolonlar kanallar ise .stack() ile (event_id, channel) satırlarına dön
            residual_tar = residual_tar.stack().to_frame(name="adc")  # index: (event_id, channel), col: adc
        else:
            # Zaten MultiIndex olabilir; tek kolona indir
            if isinstance(residual_tar, pd.DataFrame) and residual_tar.shape[1] != 1:
                residual_tar = residual.iloc[:, 0].to_frame(name="adc")
            elif isinstance(residual_tar, pd.Series):
                residual_tar = residual_tar.to_frame(name="adc")

        # 2) inputs_df'yi residual ile HİZALA (drop_constant_cm sonucu atılan kanallar inputs'tan da otomatik düşer)
        # split'i yeniden kullan
        eval_cfg_inputs = ep.EvalConfig(
            modulenames_used_for_training=[module],
            modulename_for_evaluation=module,
            nodes_per_layer=cfg.nodes_per_layer,
            dropout_rate=cfg.dropout_rate,
            modeltag=cfg.modeltag,
            inputfoldertag="",
            ncmchannels=cfg.ncmchannels,
            nch_per_erx=cfg.nch_per_erx,
        )
        io_inputs = ep.DataIO(eval_cfg_inputs)
        io_inputs.load_all()
        split_inputs = io_inputs.get_split("combined")

        inputs_df = split_inputs.inputs_df

        if isinstance(inputs_df.index, pd.MultiIndex):
            missing_in_inputs = residual.index.difference(inputs_df.index)
            if len(missing_in_inputs) > 0:
                print(f"[WARN] {len(missing_in_inputs)} rows in residual not found in inputs_df; they will be dropped.")
                residual = residual.loc[residual.index.intersection(inputs_df.index)]

            inputs_aligned = inputs_df.loc[residual.index]
        else:
            ev_ids = residual.index.get_level_values(0)
            inputs_aligned = inputs_df.loc[ev_ids].to_numpy().astype(np.float32)

        from sklearn.model_selection import train_test_split
        idx_all = residual_tar.index
        idx_train, idx_val = train_test_split(idx_all, test_size=0.20, random_state=42)

        residual_tar_train = residual_tar.loc[idx_train]
        residual_tar_val   = residual_tar.loc[idx_val]

        if isinstance(inputs_aligned, pd.DataFrame):
            inputs_train_df = inputs_aligned.loc[idx_train]
            inputs_val_df   = inputs_aligned.loc[idx_val]
            X_train = inputs_train_df.to_numpy().astype(np.float32)  # shape: (N_flat, 20)
            X_val   = inputs_val_df.to_numpy().astype(np.float32)
        else:
            ev_train = idx_train.get_level_values(0)
            ev_val   = idx_val.get_level_values(0)
            X_train = inputs_df.loc[ev_train].to_numpy().astype(np.float32)
            X_val   = inputs_df.loc[ev_val].to_numpy().astype(np.float32)

        y_train = residual_tar_train.to_numpy().astype(np.float32)  # shape: (N_flat, 1) veya (N_flat,)
        y_val   = residual_tar_val.to_numpy().astype(np.float32)

        np.save(os.path.join(dnn_output_dir, "inputs_train.npy"),  X_train)
        np.save(os.path.join(dnn_output_dir, "inputs_val.npy"),    X_val)
        np.save(os.path.join(dnn_output_dir, "targets_train.npy"), y_train)
        np.save(os.path.join(dnn_output_dir, "targets_val.npy"),   y_val)

        print(f"[OK] DNN data saved (aligned to residual index). "
              f"Final shapes: X_train {X_train.shape}, y_train {y_train.shape} | "
              f"X_val {X_val.shape}, y_val {y_val.shape}")


        # === Copy original input metadata (same as Method B) ===
        try:
            import shutil
            areimers_inputs = os.path.join(cfg.base_input_dir, module)
            dnn_inputs_dir = dnn_output_dir
            os.makedirs(dnn_inputs_dir, exist_ok=True)

            meta_files = [
                "inputs_train.npy", "inputs_val.npy", "colnames.json",
                "indices_train.npy", "indices_val.npy", "eventid.npy", "chadc.npy"
            ]
            for fname in meta_files:
                src = os.path.join(areimers_inputs, fname)
                dst = os.path.join(dnn_inputs_dir, fname)
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    print(f"[COPY] {fname} copied from Areimers → {module}/dnn_inputs")
                else:
                    print(f"[WARN] Missing {fname} in {areimers_inputs}")
        except Exception as e:
            print(f"[ERROR] Metadata copy step failed for {module}: {e}")



        # --- Step 8b: Copy missing metadata files if needed ---
        try:
            import shutil
            original_input_dir = os.path.join(cfg.base_input_dir, module)
            target_dir = dnn_output_dir

            # Define files
            computed_files = [
                "inputs_train.npy",
                "targets_train.npy",
                "inputs_val.npy",
                "targets_val.npy",
            ]
            meta_files = [
                "eventid.npy",
                "chadc.npy",
                "indices_train.npy",
                "indices_val.npy",
                #"colnames.json",
            ]

            # Check if main computation outputs exist
            if all(os.path.exists(os.path.join(target_dir, f)) for f in computed_files):
                if cfg.verbose:
                    print(f"[CHECK] Computed arrays already exist for {module}, checking metadata files...")

                for fname in meta_files:
                    dst = os.path.join(target_dir, fname)
                    src = os.path.join(original_input_dir, fname)
                    if not os.path.exists(dst):
                        if os.path.exists(src):
                            shutil.copy(src, dst)
                            print(f"[COPY] Missing metadata copied: {fname}")
                        else:
                            print(f"[WARN] Missing both source and target for {fname}")
                    else:
                        if cfg.verbose:
                            print(f"[SKIP] Metadata already exists: {fname}")
            else:
                print(f"[INFO] Computation outputs missing for {module}, skipping metadata copy check.")

        except Exception as e:
            print(f"[ERROR] Metadata copy step failed for {module}: {e}")

        # --- Step 9: Optional analytic plots
        if cfg.enable_plots:
            plot_dir = os.path.join(cfg.plots_output_dir, module, "analytic")

            if cfg.skip_existing_plots and os.path.exists(plot_dir):
                print(f"[SKIP] Plot directory already exists for {module}, skipping plots.")
            else:
                print(f"[PLOTS] Generating analytic plots for {module}...")

                # Reload if skipped earlier
                if data_exists and cfg.skip_existing_data:
                    meas_true = pd.read_pickle(os.path.join(analytic_output_dir, "meas_true.pkl"))
                    analytic_pred = pd.read_pickle(os.path.join(analytic_output_dir, "analytic_pred.pkl"))
                    residual = pd.read_pickle(os.path.join(analytic_output_dir, "residual.pkl"))
                    io = ep.DataIO(ep.EvalConfig(modulenames_used_for_training=[module],
                                                 modulename_for_evaluation=module))
                    io.load_all()
                    split = io.get_split("combined")

                variants, variants_with_cms = {
                    "true": meas_true,
                    "analytic_k0": analytic_pred
                }, {
                    "true": ep.add_cms_to_measurements_df(meas_true, split.cm_df, drop_constant_cm=False),
                    "analytic_k0": ep.add_cms_to_measurements_df(analytic_pred, split.cm_df, drop_constant_cm=False)
                }

                residuals, residuals_with_cms = {"analytic_k0": residual}, {
                    "analytic_k0": ep.add_cms_to_measurements_df(residual, split.cm_df, drop_constant_cm=False)
                }

                ep.plot_cov_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms, os.path.join(plot_dir, "covcorr"))
                ep.plot_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms, split.cm_df, os.path.join(plot_dir, "distcorr"))
                ep.plot_delta_lin_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms, split.cm_df, os.path.join(plot_dir, "delta_lin_dist_cor"))
                ep.plot_all_eigenvalues("combined", variants_with_cms, residuals_with_cms, os.path.join(plot_dir, "eigenvalues_cmincl"))
                ep.plot_all_eigenvectors(eval_cfg, "combined", variants_with_cms, residuals_with_cms, 3, os.path.join(plot_dir, "eigenvectors_cmincl"))
                ep.plot_all_projection_hists("combined", variants, residuals, split.cm_df, 3, os.path.join(plot_dir, "eigenprojections"))

                print(f"[OK] Analytic plots saved under {plot_dir}")

        # --- Step 10: DNN plots ---
        if cfg.enable_dnn_plots and (cfg.target_modules is None or module in cfg.target_modules):
            print(f"[PLOTS] Generating DNN plots for {module}...")
            model_folder_name = cfg.dnn_model_subpath.replace("dnn/dnn_models/", "").replace("/", "_")
            dnn_output_dir = os.path.join(cfg.base_output_dir, module, "dnn", "dnn_outputs", model_folder_name)

            # --- Build model-specific DNN plot directory ---
            model_folder_name = cfg.dnn_model_subpath.replace("dnn/dnn_models/", "").replace("/", "_")
            plot_dir_dnn = os.path.join(cfg.plots_output_dir, module, "dnn_predict", model_folder_name, "dnn")
            os.makedirs(plot_dir_dnn, exist_ok=True)

            meas_true_path = os.path.join(cfg.base_output_dir, module, "analytic", "meas_true.pkl")
            dnn_pred_path = os.path.join(dnn_output_dir, f"predictions_{module}_combined.pkl")

            if not (os.path.exists(meas_true_path) and os.path.exists(dnn_pred_path)):
                print(f"[SKIP] Missing DNN prediction or meas_true for {module}, skipping DNN plots.")
                continue

            meas_true = pd.read_pickle(meas_true_path)
            dnn_pred = pd.read_pickle(dnn_pred_path)
            residual_dnn = meas_true - dnn_pred

            eval_cfg = ep.EvalConfig(
                modulenames_used_for_training=[module],
                modulename_for_evaluation=module,
                nodes_per_layer=cfg.nodes_per_layer,
                dropout_rate=cfg.dropout_rate,
                modeltag=cfg.modeltag,
                inputfoldertag="",
                ncmchannels=cfg.ncmchannels,
                nch_per_erx=cfg.nch_per_erx,
            )
            io = ep.DataIO(eval_cfg)
            io.load_all()
            split = io.get_split("combined")

            variants = {"true": meas_true, "dnn": dnn_pred}
            variants_with_cms = {
                k: ep.add_cms_to_measurements_df(v, split.cm_df, drop_constant_cm=False)
                for k, v in variants.items()
            }
            residuals = {"dnn": residual_dnn}
            residuals_with_cms = {
                "dnn": ep.add_cms_to_measurements_df(residual_dnn, split.cm_df, drop_constant_cm=False)
            }

            ep.plot_cov_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                             os.path.join(plot_dir_dnn, "covcorr"))
            ep.plot_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                              split.cm_df, os.path.join(plot_dir_dnn, "distcorr"))
            ep.plot_delta_lin_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                                        split.cm_df, os.path.join(plot_dir_dnn, "delta_lin_dist_corr"))
            ep.plot_all_eigenvalues("combined", variants_with_cms, residuals_with_cms,
                                    os.path.join(plot_dir_dnn, "eigenvalues_cmincl"))
            ep.plot_all_eigenvectors(eval_cfg, "combined", variants_with_cms, residuals_with_cms,
                                     3, os.path.join(plot_dir_dnn, "eigenvectors_cmincl"))
            ep.plot_all_projection_hists("combined", variants, residuals,
                                         split.cm_df, 3, os.path.join(plot_dir_dnn, "eigenprojections"))

            print(f"[OK] DNN plots saved under {plot_dir_dnn}")


        # --- Step 11: Analytic residuals processing ---
        if cfg.enable_analytic_residuals and (cfg.analytic_residual_modules is None or module in cfg.analytic_residual_modules):
            run_analytic_residuals(cfg, module)


        # --- Step 12: Create separate noise fraction ratio plots for each method ---

        try:
            # ============================================================
            # (1) ANALYTIC METHOD
            # ============================================================
            analytic_plot_dir = os.path.join(cfg.plots_output_dir, module, "analytic", "noise_fraction_ratio")
            os.makedirs(analytic_plot_dir, exist_ok=True)

            meas_true = pd.read_pickle(os.path.join(cfg.base_output_dir, module, "analytic", "meas_true.pkl"))
            analytic_pred = pd.read_pickle(os.path.join(cfg.base_output_dir, module, "analytic", "analytic_pred.pkl"))
            residual_analytic = pd.read_pickle(os.path.join(cfg.base_output_dir, module, "analytic", "residual.pkl"))

            variants_analytic = {"true": meas_true, "analytic": analytic_pred}
            residuals_analytic = {"analytic": residual_analytic}
            """
            ep.compute_and_plot_coherent_noise(
                split_name="combined",
                cfg=cfg,
                variants=variants_analytic,
                residuals=residuals_analytic,
                plot_dir=analytic_plot_dir,
                trunc_fracs=(1.0,)
            )
            """

            variants_channel_only = {k: v.filter(regex=r"^ch_") for k, v in variants_analytic.items()}
            residuals_channel_only = {k: v.filter(regex=r"^ch_") for k, v in residuals_analytic.items()}

            ep.compute_and_plot_coherent_noise(
                split_name="combined",
                cfg=cfg,
                variants=variants_channel_only,
                residuals=residuals_channel_only,
                plot_dir=analytic_plot_dir,
                trunc_fracs=(1.0,)
            )

            print(f"[OK] Analytic noise fraction ratio saved → {analytic_plot_dir}")

            # ============================================================
            # (2) DNN METHOD
            # ============================================================
            dnn_output_dir = os.path.join(cfg.base_output_dir, module, "dnn", "dnn_outputs")
            dnn_pred_path = os.path.join(dnn_output_dir, f"predictions_{module}_combined.pkl")
            dnn_residual_path = os.path.join(dnn_output_dir, f"residual_dnn_{module}.pkl")

            if os.path.exists(dnn_pred_path) and os.path.exists(dnn_residual_path):
                dnn_plot_dir = os.path.join(cfg.plots_output_dir, module, "dnn", "model", "noise_fraction_ratio")
                os.makedirs(dnn_plot_dir, exist_ok=True)

                dnn_pred = pd.read_pickle(dnn_pred_path)
                residual_dnn = pd.read_pickle(dnn_residual_path)

                variants_dnn = {"true": meas_true, "dnn": dnn_pred}
                residuals_dnn = {"dnn": residual_dnn}
                """
                ep.compute_and_plot_coherent_noise(
                    split_name="combined",
                    cfg=cfg,
                    variants=variants_dnn,
                    residuals=residuals_dnn,
                    plot_dir=dnn_plot_dir,
                    trunc_fracs=(1.0,)
                )
                """

                variants_channel_only = {k: v.filter(regex=r"^ch_") for k, v in variants_dnn.items()}
                residuals_channel_only = {k: v.filter(regex=r"^ch_") for k, v in residuals_dnn.items()}

                ep.compute_and_plot_coherent_noise(
                    split_name="combined",
                    cfg=cfg,
                    variants=variants_channel_only,
                    residuals=residuals_channel_only,
                    plot_dir=dnn_plot_dir,
                    trunc_fracs=(1.0,)
                )

                print(f"[OK] DNN noise fraction ratio saved → {dnn_plot_dir}")
            else:
                print(f"[SKIP] No DNN results found for {module}, skipping DNN noise plots.")

            # ============================================================
            # (3) ANALYTIC RESIDUAL METHOD
            # ============================================================
            analytic_residual_dir = os.path.join(cfg.base_output_dir, module, "analytic_residual")
            corrected_path = os.path.join(analytic_residual_dir, f"residual_corrected_{module}.pkl")

            if os.path.exists(corrected_path):
                analytic_residual_plot_dir = os.path.join(cfg.plots_output_dir, module, "analytic_residual", "noise_fraction_ratio")
                os.makedirs(analytic_residual_plot_dir, exist_ok=True)

                residual_corrected = pd.read_pickle(corrected_path)

                # Variants: burada “true” olarak DNN residual veya meas_true kullanılabilir.
                # Biz DNN residual üzerinden lineer düzeltme yapıldığı için, “true” kısmı meas_true değil DNN residual olabilir.
                dnn_residual_path = os.path.join(cfg.base_output_dir, module, "dnn", "dnn_outputs", f"residual_dnn_{module}.pkl")
                if os.path.exists(dnn_residual_path):
                    residual_dnn = pd.read_pickle(dnn_residual_path)
                    variants_ar = {"true": residual_dnn, "analytic_residual": residual_corrected}
                else:
                    variants_ar = {"true": meas_true, "analytic_residual": residual_corrected}

                residuals_ar = {"analytic_residual": residual_corrected}
                """
                ep.compute_and_plot_coherent_noise(
                    split_name="combined",
                    cfg=cfg,
                    variants=variants_ar,
                    residuals=residuals_ar,
                    plot_dir=analytic_residual_plot_dir,
                    trunc_fracs=(1.0,)
                )
                """

                variants_channel_only = {k: v.filter(regex=r"^ch_") for k, v in variants_ar.items()}
                residuals_channel_only = {k: v.filter(regex=r"^ch_") for k, v in residuals_ar.items()}

                ep.compute_and_plot_coherent_noise(
                    split_name="combined",
                    cfg=cfg,
                    variants=variants_channel_only,
                    residuals=residuals_channel_only,
                    plot_dir=analytic_residual_plot_dir,
                    trunc_fracs=(1.0,)
                )

                print(f"[OK] Analytic residual noise fraction ratio saved → {analytic_residual_plot_dir}")
            else:
                print(f"[SKIP] No analytic residual results found for {module}.")

        except Exception as e:
            print(f"[WARN] Noise fraction ratio plotting failed for {module}: {e}")



    print("\n[COMPLETE] compare_methods pipeline finished successfully!\n")





def plot_combined_analytic_dnn(cfg):
    """
    Generate plots comparing analytic residuals, DNN predictions, and final residuals
    for selected modules and models in Method A.
    """
    for module in cfg.combined_modules:
        for model_layout in cfg.combined_model_layouts:
            print(f"\n[COMBINED-PLOTS] {module} | model: {model_layout}")

            base_dir = os.path.join(cfg.base_output_dir, module)
            dnn_dir = os.path.join(base_dir, "dnn", "dnn_outputs", model_layout)
            analytic_dir = os.path.join(base_dir, "analytic")

            meas_true_path = os.path.join(analytic_dir, "meas_true.pkl")
            analytic_pred_path = os.path.join(analytic_dir, "analytic_pred.pkl")
            dnn_pred_path = os.path.join(dnn_dir, f"predictions_{module}_combined.pkl")

            if not all(os.path.exists(p) for p in [meas_true_path, analytic_pred_path, dnn_pred_path]):
                print(f"[SKIP] Missing data for {module}, model {model_layout}")
                continue

            meas_true = pd.read_pickle(meas_true_path)
            analytic_pred = pd.read_pickle(analytic_pred_path)
            dnn_pred = pd.read_pickle(dnn_pred_path)

            residual_analytic = meas_true - analytic_pred
            residual_final = residual_analytic - dnn_pred

            eval_cfg = ep.EvalConfig(
                modulenames_used_for_training=[module],
                modulename_for_evaluation=module,
                nodes_per_layer=cfg.nodes_per_layer,
                dropout_rate=cfg.dropout_rate,
                modeltag=cfg.modeltag,
                inputfoldertag="",
                ncmchannels=cfg.ncmchannels,
                nch_per_erx=cfg.nch_per_erx,
            )
            io = ep.DataIO(eval_cfg)
            io.load_all()
            split = io.get_split("combined")

            variants = {
                "true": meas_true,
                "analytic_pred": analytic_pred,
                "dnn_pred": dnn_pred
            }
            residuals = {
                "analytic": residual_analytic,
                "final": residual_final
            }

            variants_with_cms = {
                k: ep.add_cms_to_measurements_df(v, split.cm_df, drop_constant_cm=False)
                for k, v in variants.items()
            }
            residuals_with_cms = {
                k: ep.add_cms_to_measurements_df(v, split.cm_df, drop_constant_cm=False)
                for k, v in residuals.items()
            }

            plot_dir = os.path.join(
                cfg.plots_output_dir, module, "dnn_outputs2", model_layout
            )
            os.makedirs(plot_dir, exist_ok=True)

            # === Correlation and eigen plots ===
            ep.plot_cov_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                             os.path.join(plot_dir, "covcorr"))
            ep.plot_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                              split.cm_df, os.path.join(plot_dir, "distcorr"))
            ep.plot_delta_lin_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                                        split.cm_df, os.path.join(plot_dir, "delta_lin_dist_corr"))
            ep.plot_all_eigenvalues("combined", variants_with_cms, residuals_with_cms,
                                    os.path.join(plot_dir, "eigenvalues_cmincl"))
            ep.plot_all_eigenvectors(eval_cfg, "combined", variants_with_cms, residuals_with_cms,
                                     3, os.path.join(plot_dir, "eigenvectors_cmincl"))
            ep.plot_all_projection_hists("combined", variants, residuals,
                                         split.cm_df, 3, os.path.join(plot_dir, "eigenprojections"))

            # === Noise fraction ratio ===
            variants_channel_only = {k: v.filter(regex=r"^ch_") for k, v in variants.items()}
            residuals_channel_only = {k: v.filter(regex=r"^ch_") for k, v in residuals.items()}

            ep.compute_and_plot_coherent_noise(
                "combined",
                eval_cfg,
                variants_channel_only,
                residuals_channel_only,
                os.path.join(plot_dir, "noise_fraction_ratio")
            )

            print(f"[OK] Combined analytic + DNN plots saved → {plot_dir}")


def compute_coherent_noise_methodA(cfg):
    """
    Method A coherent noise evaluation:
      residual_analytic  = meas_true - analytic_pred  (from ./method_A/<module>/analytic/residual.pkl)
      final_residual     = residual_analytic - dnn_pred
    Computes:
      - coherent noise ratio (HGCal definition: rms_direct² − rms_alternating²)
      - global RMS improvement = RMS(before)/RMS(after)
    Optionally draws coherent noise plots if cfg.enable_coherent_noise_plots is True.
    """

    import os
    import numpy as np
    import pandas as pd

    # --- 0. enable flags ---
    if not getattr(cfg, "enable_coherent_noise_eval", False):
        print("[SKIP] Coherent noise ratio evaluation disabled in config.")
        return

    # --- 1. utilities ---
    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")

    def _compute_coh_noise_ratio_from_block(block: np.ndarray) -> float:
        if block.ndim != 2 or block.shape[1] == 0:
            return float("nan")
        n_channels = block.shape[1]
        direct_sum = block.sum(axis=1)
        alternating_sum = block[:, ::2].sum(axis=1) - block[:, 1::2].sum(axis=1)
        rms_direct = np.sqrt(np.mean(direct_sum ** 2))
        rms_alternating = np.sqrt(np.mean(alternating_sum ** 2))
        delta = rms_direct**2 - rms_alternating**2
        sigma_incoherent = rms_alternating / np.sqrt(n_channels)
        sigma_coherent = np.sign(delta) * np.sqrt(abs(delta)) / n_channels
        if sigma_incoherent == 0:
            return np.nan
        return sigma_coherent / sigma_incoherent

    def _align_cols(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        common_idx = df_a.index.intersection(df_b.index)
        if len(common_idx) == 0:
            raise ValueError("No overlapping indices between analytic residual and DNN prediction.")
        ch_cols_a = [c for c in df_a.columns if isinstance(c, str) and c.startswith("ch_")]
        ch_cols_b = set([c for c in df_b.columns if isinstance(c, str) and c.startswith("ch_")])
        common_cols = [c for c in ch_cols_a if c in ch_cols_b]
        if len(common_cols) == 0:
            raise ValueError("No overlapping channel columns (regex '^ch_').")
        a2 = df_a.loc[common_idx, common_cols].sort_index()
        b2 = df_b.loc[common_idx, common_cols].sort_index()
        return a2, b2

    def _chunk_erx(columns: list[str], nch_per_erx: int) -> list[list[str]]:
        def chnum(c):
            try:
                return int(c.split("_")[1])
            except Exception:
                return 10**9
        cols_sorted = sorted(columns, key=chnum)
        groups = [cols_sorted[i:i + max(1, int(nch_per_erx))]
                  for i in range(0, len(cols_sorted), max(1, int(nch_per_erx)))]
        return groups

    # --- 2. determine which modules to run ---
    modules = getattr(cfg, "coherent_noise_modules", None) or \
              getattr(cfg, "combined_modules", None) or \
              getattr(cfg, "analytic_residual_modules", None)

    if not modules:
        modules = [m for m in sorted(os.listdir(cfg.base_input_dir))
                   if os.path.isdir(os.path.join(cfg.base_input_dir, m))]

    model_layouts = getattr(cfg, "combined_model_layouts", [])
    if isinstance(model_layouts, str):
        model_layouts = [model_layouts]
    if not model_layouts:
        print("[INFO] No combined_model_layouts defined; nothing to compute.")
        return

    # --- 3. main loop ---
    for module in modules:
        analytic_dir = os.path.join(cfg.base_output_dir, module, "analytic")
        residual_path = os.path.join(analytic_dir, "residual.pkl")
        if not os.path.exists(residual_path):
            print(f"[SKIP] Missing analytic residual for {module}: {residual_path}")
            continue

        try:
            residual_analytic = pd.read_pickle(residual_path)
        except Exception as e:
            print(f"[SKIP] Failed to read residual.pkl for {module}: {e}")
            continue

        results_path = os.path.join(cfg.base_output_dir, module, "results_method_A.txt")
        header_needed = not os.path.exists(results_path)

        for model_layout in model_layouts:
            dnn_dir = os.path.join(cfg.base_output_dir, module, "dnn", "dnn_outputs", model_layout)
            dnn_pred_path = os.path.join(dnn_dir, f"predictions_{module}_combined.pkl")
            use_with_cm = False
            if not os.path.exists(dnn_pred_path):
                dnn_pred_with_cm_path = os.path.join(dnn_dir, f"predictions_{module}_withCM_combined.pkl")
                if os.path.exists(dnn_pred_with_cm_path):
                    dnn_pred_path = dnn_pred_with_cm_path
                    use_with_cm = True
                else:
                    print(f"[SKIP] Missing DNN prediction for {module} | {model_layout}")
                    continue

            try:
                dnn_pred = pd.read_pickle(dnn_pred_path)
                if use_with_cm:
                    dnn_pred = dnn_pred.filter(regex=r"^ch_")
            except Exception as e:
                print(f"[SKIP] Failed to read DNN prediction ({dnn_pred_path}): {e}")
                continue

            residual_analytic_ch = residual_analytic.filter(regex=r"^ch_")
            if residual_analytic_ch.empty:
                print(f"[SKIP] No channel columns in analytic residual for {module}.")
                continue

            try:
                before_df, dnn_df = _align_cols(residual_analytic_ch, dnn_pred)
            except Exception as e:
                print(f"[SKIP] Alignment failed for {module} | {model_layout}: {e}")
                continue

            # --- compute corrected residual ---
            after_df = before_df - dnn_df

            # --- global RMS improvement ---
            rms_b = _rms(before_df.to_numpy().ravel())
            rms_a = _rms(after_df.to_numpy().ravel())
            frac_impr = (rms_b / rms_a) if (rms_a > 0 and np.isfinite(rms_a)) else float("nan")

            # --- coherent noise ratio ---
            erx_groups = _chunk_erx(list(before_df.columns), getattr(cfg, "nch_per_erx", 37))
            ratios = []
            for grp in erx_groups:
                b = before_df[grp].to_numpy()
                a = after_df[grp].to_numpy()
                coh_b = _compute_coh_noise_ratio_from_block(b)
                coh_a = _compute_coh_noise_ratio_from_block(a)
                if np.isfinite(coh_b) and coh_b != 0:
                    ratios.append(coh_a / coh_b)
            coh_ratio_mean = float(np.nanmean(ratios)) if ratios else float("nan")

            # --- write numeric result ---
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "a", encoding="utf-8") as f:
                if header_needed:
                    f.write("module,model_layout,dropout,nch_per_erx,rms_before,rms_after,frac_improvement,coh_ratio_mean\n")
                    header_needed = False
                f.write(f"{module},{model_layout},{getattr(cfg,'dropout_rate',0.0)},{getattr(cfg,'nch_per_erx',37)},"
                        f"{rms_b:.6f},{rms_a:.6f},{frac_impr:.6f},{coh_ratio_mean:.6f}\n")

            print(f"[OK] Method A coherent noise → {module} | {model_layout} | "
                  f"FracImprovement={frac_impr:.4f} (↑ iyi), CohRatio={coh_ratio_mean:.4f} (↓ iyi)")

            # --- plot generation (if enabled) ---
            if getattr(cfg, "enable_coherent_noise_plots", False):
                try:
                    from evaluate_performance import infer_erx_groups_from_columns, compute_and_plot_coherent_noise
                    plot_dir = os.path.join(cfg.base_output_dir, module, "plots", "noise_correction")
                    os.makedirs(plot_dir, exist_ok=True)
                    variants = {"true": before_df}
                    residuals = {"methodA": after_df}
                    compute_and_plot_coherent_noise(
                        split_name=f"{module}_{model_layout}",
                        cfg=cfg,
                        variants=variants,
                        residuals=residuals,
                        plot_dir=plot_dir,
                        trunc_fracs=(1.0,)
                    )
                except Exception as e:
                    print(f"[WARN] Coherent noise plot skipped for {module} | {model_layout}: {e}")


def plot_combined_method_B_dnn(cfg):
    """
    Generate combined plots for Method B:
    - 'true' = residuals from first DNN
    - 'dnn_pred' = predictions from second DNN
    - 'residual' = true - dnn_pred
    Plots saved under:
    plots/performance/trained_<trained_module>/evaluated_<evaluated_module>/dnn_output3/<model_layout>/
    """
    trained_module = cfg.method_B_trained_module
    evaluated_module = cfg.method_B_evaluated_module

    for model_layout in cfg.method_B_combined_models:
        print(f"\n[COMBINED-METHOD-B] {trained_module} → {evaluated_module} | model: {model_layout}")

        # === Define base directories ===
        base_dir = os.path.join(
            "./method_B",
            f"trained_{trained_module}",
            f"evaluated_{evaluated_module}",
            "dnn"
        )
        dnn_dir = os.path.join(base_dir, "dnn_outputs2", model_layout)
        residual1_path = os.path.join(base_dir, "dnn_output", cfg.method_B_model_subpath, "residual.pkl")

        # === Check required files ===
        dnn_pred_path = os.path.join(dnn_dir, "pred.pkl")

        if not all(os.path.exists(p) for p in [residual1_path, dnn_pred_path]):
            print(f"[SKIP] Missing data for model {model_layout} → {evaluated_module}")
            print(f"       residual1: {os.path.exists(residual1_path)}, dnn_pred: {os.path.exists(dnn_pred_path)}")
            continue

        # === Load data ===
        residual1_df = pd.read_pickle(residual1_path)   # True = 1st DNN residual
        dnn_pred_df = pd.read_pickle(dnn_pred_path)     # 2nd DNN prediction
        print(f"[INFO] Loaded residual1_df {residual1_df.shape}, dnn_pred_df {dnn_pred_df.shape}")

        # === Align indices ===
        if not residual1_df.index.equals(dnn_pred_df.index):
            common_index = residual1_df.index.intersection(dnn_pred_df.index)
            residual1_df = residual1_df.loc[common_index]
            dnn_pred_df = dnn_pred_df.loc[common_index]

        # === Compute final residual ===
        residual_final_df = residual1_df - dnn_pred_df

        # === Eval config ===
        eval_cfg = ep.EvalConfig(
            modulenames_used_for_training=[evaluated_module],
            modulename_for_evaluation=evaluated_module,
            nodes_per_layer=cfg.nodes_per_layer,
            dropout_rate=cfg.dropout_rate,
            modeltag=cfg.modeltag,
            inputfoldertag="",
            ncmchannels=cfg.ncmchannels,
            nch_per_erx=cfg.nch_per_erx,
        )
        io = ep.DataIO(eval_cfg)
        io.load_all()
        split = io.get_split("combined")

        # === Prepare variant dicts ===
        variants = {
            "true": residual1_df,
            "dnn_pred": dnn_pred_df
        }
        residuals = {
            "final": residual_final_df
        }

        variants_with_cms = {
            k: ep.add_cms_to_measurements_df(v, split.cm_df, drop_constant_cm=False)
            for k, v in variants.items()
        }
        residuals_with_cms = {
            k: ep.add_cms_to_measurements_df(v, split.cm_df, drop_constant_cm=False)
            for k, v in residuals.items()
        }

        # === Create output dir ===
        plot_dir = os.path.join(
            cfg.plots_output_dir,
            f"trained_{trained_module}",
            f"evaluated_{evaluated_module}",
            "dnn_output3",
            model_layout
        )
        os.makedirs(plot_dir, exist_ok=True)

        # === Core plots ===
        ep.plot_cov_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                         os.path.join(plot_dir, "covcorr"))
        ep.plot_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                          split.cm_df, os.path.join(plot_dir, "distcorr"))
        ep.plot_delta_lin_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                                    split.cm_df, os.path.join(plot_dir, "delta_lin_dist_corr"))
        ep.plot_all_eigenvalues("combined", variants_with_cms, residuals_with_cms,
                                os.path.join(plot_dir, "eigenvalues_cmincl"))
        ep.plot_all_eigenvectors(eval_cfg, "combined", variants_with_cms, residuals_with_cms,
                                 3, os.path.join(plot_dir, "eigenvectors_cmincl"))
        ep.plot_all_projection_hists("combined", variants, residuals,
                                     split.cm_df, 3, os.path.join(plot_dir, "eigenprojections"))

        # === Noise fraction ratio ===
        variants_channel_only = {k: v.filter(regex=r"^ch_") for k, v in variants.items()}
        residuals_channel_only = {k: v.filter(regex=r"^ch_") for k, v in residuals.items()}

        ep.compute_and_plot_coherent_noise(
            "combined",
            eval_cfg,
            variants_channel_only,
            residuals_channel_only,
            os.path.join(plot_dir, "noise_fraction_ratio")
        )

        print(f"[OK] Method B combined plots saved → {plot_dir}")


def run_analytic_residuals(cfg, module):
    """
    Perform analytic inference on residuals obtained from DNN predictions.
    Results will be stored under:
      method_A/<module>/analytic_residual/
    """

    print(f"\n[ANALYTIC-RESIDUAL] Processing module: {module}")

    # --- Define correct local paths ---
    module_root = os.path.join(cfg.base_output_dir, module)
    analytic_residual_dir = os.path.join(module_root, "analytic_residual")
    os.makedirs(analytic_residual_dir, exist_ok=True)

    # Input residual from DNN predictions
    dnn_residual_path = os.path.join(
        cfg.base_output_dir, module, "dnn", "dnn_outputs", f"residual_dnn_{module}.pkl"
    )
    if not os.path.exists(dnn_residual_path):
        print(f"[SKIP] Missing DNN residual for {module}, skipping analytic residual computation.")
        return

    residual_df = pd.read_pickle(dnn_residual_path)

    # Load CM reference
    eval_cfg = ep.EvalConfig(
        modulenames_used_for_training=[module],
        modulename_for_evaluation=module,
        nodes_per_layer=cfg.nodes_per_layer,
        dropout_rate=cfg.dropout_rate,
        modeltag=cfg.modeltag,
        inputfoldertag="",
        ncmchannels=cfg.ncmchannels,
        nch_per_erx=cfg.nch_per_erx,
    )
    io = ep.DataIO(eval_cfg)
    io.load_all()
    split = io.get_split("combined")

    # --- Run analytic inference on residuals ---
    analytic = ep.AnalyticInferencer(drop_constant_cm=cfg.drop_constant_cm)
    analytic.fit(split)
    analytic_pred_residual = analytic.predict(split_predict=split, split_correction=split)

    residual_corrected = residual_df - analytic_pred_residual

    # --- Save outputs under module/analytic_residual ---
    residual_df.to_pickle(os.path.join(analytic_residual_dir, f"residual_input_{module}.pkl"))
    analytic_pred_residual.to_pickle(os.path.join(analytic_residual_dir, f"analytic_pred_residual_{module}.pkl"))
    residual_corrected.to_pickle(os.path.join(analytic_residual_dir, f"residual_corrected_{module}.pkl"))

    print(f"[OK] Analytic residual outputs saved → {analytic_residual_dir}")

    # --- Generate plots ---
    plot_dir = os.path.join(cfg.plots_output_dir, module, "analytic_residual")
    os.makedirs(plot_dir, exist_ok=True)

    # Only keep relevant variants to avoid extra folders
    variants = {
        "true": residual_df,
        "analytic": analytic_pred_residual
    }
    variants_with_cms = {
        k: ep.add_cms_to_measurements_df(v, split.cm_df, drop_constant_cm=False)
        for k, v in variants.items()
    }

    residuals = {"analytic": residual_corrected}
    residuals_with_cms = {
        "analytic": ep.add_cms_to_measurements_df(residual_corrected, split.cm_df, drop_constant_cm=False)
    }

    ep.plot_cov_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                     os.path.join(plot_dir, "covcorr"))
    ep.plot_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                      split.cm_df, os.path.join(plot_dir, "distcorr"))
    ep.plot_delta_lin_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                                split.cm_df, os.path.join(plot_dir, "delta_lin_dist_corr"))
    ep.plot_all_eigenvalues("combined", variants_with_cms, residuals_with_cms,
                            os.path.join(plot_dir, "eigenvalues_cmincl"))
    ep.plot_all_eigenvectors(eval_cfg, "combined", variants_with_cms, residuals_with_cms,
                             3, os.path.join(plot_dir, "eigenvectors_cmincl"))
    ep.plot_all_projection_hists("combined", variants, residuals,
                                 split.cm_df, 3, os.path.join(plot_dir, "eigenprojections"))

    print(f"[OK] Analytic residual plots saved under {plot_dir}\n")

def plot_loss_for_module(cfg, module):
    """
    Loads and visualizes the DNN loss curves (train & validation)
    for the given module using the configuration settings.
    """
    # Path to DNN model directory containing val_losses.npy
    model_folder = os.path.join(cfg.base_output_dir, module, cfg.dnn_model_subpath)
    val_losses_path = os.path.join(model_folder, "val_losses.npy")
    
    if not os.path.exists(val_losses_path):
        print(f"[ERROR] {val_losses_path} does not exist for {module}")
        return

    # Directory where the plot will be saved
    plot_dir = os.path.join(cfg.plots_output_dir, module, "performance_loss")
    os.makedirs(plot_dir, exist_ok=True)

    # Call the evaluate_performance.plot_loss function correctly
    print(f"[INFO] Plotting loss curve for module: {module}")
    ep.plot_loss(model_folder, plot_dir)
    print(f"[OK] Loss plot saved → {plot_dir}")


# Function to generate performance plots if enabled
def generate_performance_plots(cfg):
    """Function used to generate performance plots"""
    if cfg.enable_performance_plots:
        for module in cfg.enabled_modules_for_plots:
            plot_loss_for_module(cfg, module)
    else:
        print("[INFO] Performance plots are disabled.")



def run_method_B(cfg):

    import os, json, shutil
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split

    from models import DNNFlex
    import evaluate_performance as ep
    from evaluate_performance import pivot_flat_preds_to_event_channel

    trained_module   = cfg.method_B_trained_module_dnn1
    evaluated_module = cfg.method_B_evaluated_module_dnn1

    print(f"\n[METHOD B] DNN1: {trained_module} → {evaluated_module}")

    # ================================
    # === DIRECTORIES
    # ================================
    areimers_inputs = os.path.join(cfg.base_input_dir, evaluated_module)
    print(f"[DEBUG] Using AREIMERS input folder: {areimers_inputs}")

    trained_model_dir = os.path.join(
        "/eos/user/a/areimers/hgcal/dnn_models",
        trained_module, cfg.method_B_model_subpath
    )
    weights_path = os.path.join(trained_model_dir, "regression_dnn_best.pth")

    # === OVERRIDE DNN1 MODEL WEIGHTS FOR THIS RUN ===
    override_model_dir = "/eos/user/g/gmihriye/CM/compare_methods/CMCorrection/control_models/ML_F3W_WXIH0191/in20__512-512-512-512-64__dr0"
    weights_path = os.path.join(override_model_dir, "regression_dnn_best.pth")

    print(f"[OVERRIDE] Using custom DNN1 weights: {weights_path}")


    print(f"[DEBUG] DNN1 model dir: {trained_model_dir}")
    print(f"[DEBUG] DNN1 weights path: {weights_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[ERROR] DNN1 weights not found: {weights_path}")

    base_output_dir = os.path.join(
        "./method_B",
        f"trained_{trained_module}",
        f"evaluated_{evaluated_module}",
        "dnn"
    )
    dnn_inputs_dir = os.path.join(base_output_dir, "dnn_inputs")
    os.makedirs(dnn_inputs_dir, exist_ok=True)

    # ================================
    # === LOAD INPUTS (train+val)
    # ================================
    inputs_train = np.load(os.path.join(areimers_inputs, "inputs_train.npy"))
    inputs_val   = np.load(os.path.join(areimers_inputs, "inputs_val.npy"))
    inputs_combined = np.concatenate([inputs_train, inputs_val], axis=0)

    print(f"[INFO] Loaded Method B combined inputs: {inputs_combined.shape}")

    # Sanity check with colnames.json
    colnames_path = os.path.join(areimers_inputs, "colnames.json")
    with open(colnames_path, "r") as f:
        colnames = json.load(f)

    print(f"[DEBUG] Number of colnames = {len(colnames)}")
    if inputs_combined.shape[1] != len(colnames):
        raise RuntimeError(
            f"[ERROR] inputs_combined has {inputs_combined.shape[1]} features but "
            f"colnames.json has {len(colnames)} entries."
        )

    # ================================
    # === PARSE ARCHITECTURE
    # ================================
    ms = cfg.method_B_model_subpath
    try:
        if ms.startswith("in") and "__" in ms:
            # Example: "in20__512-512-512-512-64__dr0"
            parts = ms.split("__")
            in_part = parts[0]      # e.g. "in20"
            arch_str = parts[1]     # e.g. "512-512-512-512-64"

            # expected input dim from name, e.g. "in20" -> 20
            try:
                expected_in_dim = int(in_part.replace("in", ""))
            except ValueError:
                expected_in_dim = None

            layer_sizes = [int(x) for x in arch_str.split("-") if x.isdigit()]
        else:
            expected_in_dim = None
            layer_sizes = cfg.nodes_per_layer
    except Exception as e:
        print(f"[WARN] Could not parse architecture from method_B_model_subpath='{ms}': {e}")
        expected_in_dim = None
        layer_sizes = cfg.nodes_per_layer

    print(f"[INFO] Method B DNN1 architecture from path: {layer_sizes}")
    if expected_in_dim is not None:
        print(f"[DEBUG] Expected DNN1 input dim from name: {expected_in_dim}")
        if expected_in_dim != inputs_combined.shape[1]:
            print(
                f"[WARN] Expected DNN1 input dim {expected_in_dim} from name, "
                f"but inputs_combined.shape[1] = {inputs_combined.shape[1]}"
            )

    # ================================
    # === RUN DNN1 INFERENCE
    # ================================
    model = DNNFlex(inputs_combined.shape[1], layer_sizes, cfg.dropout_rate)
    print(f"[DEBUG] Created DNNFlex with input_dim={inputs_combined.shape[1]}, "
          f"layers={layer_sizes}, dropout={cfg.dropout_rate}")

    # Extra debug: print first layer weight shape after loading
    inferencer = ep.DNNInferencer(
        model=model,
        weights_path=weights_path,
        dtype=np.float32,
        batch_size=16384
    )
    first_param = next(model.parameters())
    print(f"[DEBUG] First layer weight shape after loading state_dict: {first_param.shape}")

    preds_flat = inferencer(inputs_combined)
    print(f"[CHECK] preds_flat shape: {preds_flat.shape}")
    print(f"[DEBUG] preds_flat stats: mean={preds_flat.mean():.5f}, std={preds_flat.std():.5f}, "
          f"min={preds_flat.min():.5f}, max={preds_flat.max():.5f}")

    # ================================
    # === LOAD SPLIT AND MEASUREMENTS
    # ================================
    eval_cfg_inputs = ep.EvalConfig(
        modulenames_used_for_training=[evaluated_module],
        modulename_for_evaluation=evaluated_module,
        nodes_per_layer=cfg.nodes_per_layer,
        dropout_rate=cfg.dropout_rate,
        modeltag=cfg.modeltag,
        inputfoldertag="",
        nch_per_erx=cfg.nch_per_erx,
        ncmchannels=cfg.ncmchannels,
    )

    io = ep.DataIO(eval_cfg_inputs)
    io.load_all()
    split = io.get_split("combined")

    print(f"[DEBUG] split.targets_flat shape: {split.targets_flat.shape}")
    print(f"[DEBUG] split.inputs_flat  shape: {split.inputs_flat.shape}")
    print(f"[DEBUG] split.measurements_df shape: {split.measurements_df.shape}")
    print(f"[DEBUG] split.eventid_flat len: {len(split.eventid_flat)}, "
          f"split.channels_flat len: {len(split.channels_flat)}")

    # Sanity: number of samples must match
    if preds_flat.shape[0] != split.targets_flat.shape[0]:
        raise RuntimeError(
            f"[ERROR] DNN1 returned {preds_flat.shape[0]} preds, "
            f"but combined split has {split.targets_flat.shape[0]} samples."
        )

    # ================================
    # === preds_df pivot format (event x channel)
    # ================================
    preds_df = pivot_flat_preds_to_event_channel(
        preds_flat, split.eventid_flat, split.channels_flat, split.measurements_df
    )
    print(f"[CHECK] preds_df ready: shape={preds_df.shape}")
    print(f"[DEBUG] preds_df stats: mean={preds_df.values.mean():.5f}, "
          f"std={preds_df.values.std():.5f}")



    # ================================
    # === MEAS TRUE → AUTO-DETECT FORMAT
    # ================================
    meas_true = split.measurements_df.copy()

    if meas_true.index.nlevels == 1:
        print("[INFO] meas_true is already pivot format (event x channel)")
        meas_true_pivot = meas_true.copy()
    else:
        print("[INFO] meas_true is MultiIndex → converting to pivot")
        meas_true.index.set_names(["event","channel"], inplace=True)
        meas_true_pivot = meas_true.unstack(level="channel")

    print(f"[CHECK] meas_true_pivot shape: {meas_true_pivot.shape}")
    print(f"[DEBUG] meas_true stats: mean={meas_true_pivot.values.mean():.5f}, "
          f"std={meas_true_pivot.values.std():.5f}")

    print("[DEBUG] preds_df columns (first 20):", preds_df.columns[:20].tolist())
    print("[DEBUG] meas_true_pivot columns (first 20):", meas_true_pivot.columns[:20].tolist())


    # Extra consistency checks: same index/columns
    if not meas_true_pivot.index.equals(preds_df.index):
        raise RuntimeError("[ERROR] meas_true_pivot and preds_df have different event indices.")
    if not meas_true_pivot.columns.equals(preds_df.columns):
        raise RuntimeError("[ERROR] meas_true_pivot and preds_df have different channel columns.")

    # ================================
    # === RESIDUAL = true - pred
    # ================================
    residual_pivot = meas_true_pivot - preds_df
    print(f"[CHECK] residual_pivot shape: {residual_pivot.shape}")
    print(f"[DEBUG] residual stats: mean={residual_pivot.values.mean():.5f}, "
          f"std={residual_pivot.values.std():.5f}, "
          f"min={residual_pivot.values.min():.5f}, "
          f"max={residual_pivot.values.max():.5f}")

    # ================================
    # === SAVE DNN1 true / pred / residual PKL FILES ===
    # ================================
    dnn1_output_dir = os.path.join(
        "./method_B",
        f"trained_{trained_module}",
        f"evaluated_{evaluated_module}",
        "dnn",
        "dnn1_outputs",
        cfg.method_B_model_subpath
    )
    os.makedirs(dnn1_output_dir, exist_ok=True)

    meas_true_pivot.to_pickle(os.path.join(dnn1_output_dir, "true.pkl"))
    preds_df.to_pickle(os.path.join(dnn1_output_dir, "pred.pkl"))
    residual_pivot.to_pickle(os.path.join(dnn1_output_dir, "residual.pkl"))

    print(f"[OK] Saved DNN1 PKLs → {dnn1_output_dir}")


    # Optionally, check a few channels
    for ch_name in list(meas_true_pivot.columns)[:3]:
        t = meas_true_pivot[ch_name].to_numpy()
        p = preds_df[ch_name].to_numpy()
        r = residual_pivot[ch_name].to_numpy()
        print(f"[DEBUG] Channel {ch_name}: "
              f"true std={np.std(t):.5f}, pred std={np.std(p):.5f}, "
              f"resid std={np.std(r):.5f}")

    # ================================
    # === FLATTEN RESIDUAL TO (event, channel_int)
    # ================================
    residual_flat = residual_pivot.stack().to_frame(name="adc")
    residual_flat.index.set_names(["event", "channel"], inplace=True)

    ev = residual_flat.index.get_level_values("event")
    ch = residual_flat.index.get_level_values("channel")

    if isinstance(ch[0], str):
        ch_int = ch.map(lambda x: int(str(x).split("_")[1]))
    else:
        ch_int = ch.astype(int)

    residual_flat.index = pd.MultiIndex.from_arrays(
        [ev.astype(int), ch_int],
        names=["event", "channel"],
    )

    print(f"[CHECK] residual_flat shape: {residual_flat.shape}")


    print("[DEBUG] residual_flat index (first 10):", list(residual_flat.index[:10]))

    # ================================
    # === PREPARE INPUTS_DF (same index)
    # ================================
    # We already loaded colnames above
    inputs_df = pd.DataFrame(inputs_combined, columns=colnames)

    inputs_df.index = pd.MultiIndex.from_arrays(
        [
            split.eventid_flat.astype(int),
            split.channels_flat.astype(int),
        ],
        names=["event", "channel"],
    )

    print(f"[CHECK] inputs_df ready: shape={inputs_df.shape}")
    print(f"[DEBUG] inputs_df columns (first 10): {inputs_df.columns[:10].tolist()}")


    # ================================
    # === METHOD A STYLE ALIGNMENT
    # ================================
    # residual_flat.index → ana referans (Method A böyle yapıyor)

    residual_index = residual_flat.index

    # 1) inputs_df residual index'i TAM olarak içermeli
    if not residual_index.isin(inputs_df.index).all():
        missing = residual_index.difference(inputs_df.index)
        raise RuntimeError(
            f"[ERROR] inputs_df is missing {len(missing)} rows "
            f"that appear in residual_flat — Method B must match Method A."
        )

    # 2) EXACT same order
    inputs_aligned  = inputs_df.loc[residual_index]
    residual_aligned = residual_flat.loc[residual_index]

    print("[INFO] Method-A-style alignment applied:")
    print(" inputs_aligned shape =", inputs_aligned.shape)
    print(" targets_aligned shape =", residual_aligned.shape)


    # ================================
    # === TRAIN/VAL SPLIT
    # ================================
    idx_train, idx_val = train_test_split(residual_index, test_size=0.20, random_state=42)


    X_train = inputs_aligned.loc[idx_train].to_numpy(np.float32)
    X_val   = inputs_aligned.loc[idx_val].to_numpy(np.float32)
    y_train = residual_aligned.loc[idx_train]["adc"].to_numpy(np.float32).reshape(-1, 1)
    y_val   = residual_aligned.loc[idx_val]["adc"].to_numpy(np.float32).reshape(-1, 1)

    print("[FIX] y_train fixed shape:", y_train.shape)
    print("[FIX] y_val fixed shape:", y_val.shape)

    print(f"[DEBUG] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"[DEBUG] X_val   shape: {X_val.shape},   y_val   shape: {y_val.shape}")
    print(f"[DEBUG] y_train stats: mean={y_train.mean():.5f}, std={y_train.std():.5f}")
    print(f"[DEBUG] y_val   stats: mean={y_val.mean():.5f}, std={y_val.std():.5f}")

    # ================================
    # === SAVE Method-A-style OUTPUTS
    # ================================
    np.save(os.path.join(dnn_inputs_dir, "inputs_train.npy"),  X_train)
    np.save(os.path.join(dnn_inputs_dir, "inputs_val.npy"),    X_val)
    np.save(os.path.join(dnn_inputs_dir, "targets_train.npy"), y_train)
    np.save(os.path.join(dnn_inputs_dir, "targets_val.npy"),   y_val)

    common_idx = residual_aligned.index

    np.save(os.path.join(dnn_inputs_dir, "eventid.npy"), common_idx.get_level_values(0))
    np.save(os.path.join(dnn_inputs_dir, "chadc.npy"),   common_idx.get_level_values(1))

    np.save(os.path.join(dnn_inputs_dir, "indices_train.npy"), np.arange(len(idx_train)))
    np.save(os.path.join(dnn_inputs_dir, "indices_val.npy"),   np.arange(len(idx_val)))

    print(f"[OK] Method B complete → DNN2 inputs saved in {dnn_inputs_dir}")

    # ================================
    # === COPY COLNAMES.JSON ===
    # ================================
    colnames_src = os.path.join(areimers_inputs, "colnames.json")
    colnames_dst = os.path.join(dnn_inputs_dir, "colnames.json")

    try:
        shutil.copy(colnames_src, colnames_dst)
        print(f"[OK] Copied colnames.json → {colnames_dst}")
    except Exception as e:
        print(f"[WARN] Could not copy colnames.json: {e}")

    # ================================
    # === AUTO-PLOT DNN1 OUTPUTS ===
    # ================================
    if getattr(cfg, "method_B_enable_plots_dnn1", False):
        print("[INFO] Plotting DNN1 outputs for Method B…")
        plot_methodB_dnn1(cfg)
    else:
        print("[INFO] Method B plots disabled.")



def plot_methodB_dnn1(cfg):
    """
    Method B – DNN1 için Method A'daki DNN plotlarına birebir benzer plotlar üretir.
    """
    import os
    import pandas as pd
    import evaluate_performance as ep

    trained_module   = cfg.method_B_trained_module_dnn1
    evaluated_module = cfg.method_B_evaluated_module_dnn1
    model_layout     = cfg.method_B_model_subpath

    print(f"\n[PLOTS - METHOD B DNN1] {trained_module} → {evaluated_module}")

    # === Load saved true/pred/residual files ===
    dnn1_output_dir = os.path.join(
        "./method_B",
        f"trained_{trained_module}",
        f"evaluated_{evaluated_module}",
        "dnn",
        "dnn1_outputs",
        model_layout
    )

    true_path     = os.path.join(dnn1_output_dir, "true.pkl")
    pred_path     = os.path.join(dnn1_output_dir, "pred.pkl")
    residual_path = os.path.join(dnn1_output_dir, "residual.pkl")

    if not (os.path.exists(true_path) and os.path.exists(pred_path) and os.path.exists(residual_path)):
        print("[SKIP] Missing DNN1 outputs for Method B — cannot generate plots.")
        return

    meas_true  = pd.read_pickle(true_path)
    dnn_pred   = pd.read_pickle(pred_path)
    residual   = pd.read_pickle(residual_path)

    # --- Eval config to load CM df ---
    eval_cfg = ep.EvalConfig(
        modulenames_used_for_training=[evaluated_module],
        modulename_for_evaluation=evaluated_module,
        nodes_per_layer=cfg.nodes_per_layer,
        dropout_rate=cfg.dropout_rate,
        modeltag=cfg.modeltag,
        inputfoldertag="",
        ncmchannels=cfg.ncmchannels,
        nch_per_erx=cfg.nch_per_erx,
    )

    io = ep.DataIO(eval_cfg)
    io.load_all()
    split = io.get_split("combined")

    # === Method A style dicts ===
    variants = {"true": meas_true, "dnn": dnn_pred}
    residuals = {"dnn": residual}

    variants_with_cms = {
        k: ep.add_cms_to_measurements_df(v, split.cm_df, drop_constant_cm=False)
        for k, v in variants.items()
    }

    residuals_with_cms = {
        "dnn": ep.add_cms_to_measurements_df(residual, split.cm_df, drop_constant_cm=False)
    }

    # === Plot directory ===
    plot_dir = os.path.join(
        "plots/performance",
        f"trained_{trained_module}",
        f"evaluated_{evaluated_module}",
        model_layout,
        "dnn_plots_v1"
    )
    os.makedirs(plot_dir, exist_ok=True)

    # --- Ensure all subfolders required by evaluate_performance exist ---
    required_subdirs = [
        "covcorr",
        "distcorr",
        "delta_lin_dist_corr",
        "coherent_noise",
        "eigenvalues",
        "eigenvectors",
        "eigenvectors/eigenvectors_2d",
        "eigenprojections",
    ]

    for sd in required_subdirs:
        os.makedirs(os.path.join(plot_dir, sd), exist_ok=True)

    print(f"[INFO] Method B DNN1 plots will be saved under: {plot_dir}")

    # ============================================================
    # 1) Heatmaps (CM INCLUDED)
    # ============================================================
    ep.plot_cov_corr(
        "combined", eval_cfg,
        variants_with_cms, residuals_with_cms,
        os.path.join(plot_dir, "covcorr")
    )

    ep.plot_dist_corr(
        "combined", eval_cfg,
        variants_with_cms, residuals_with_cms,
        split.cm_df, os.path.join(plot_dir, "distcorr")
    )

    ep.plot_delta_lin_dist_corr(
        "combined", eval_cfg,
        variants_with_cms, residuals_with_cms,
        split.cm_df, os.path.join(plot_dir, "delta_lin_dist_corr")
    )

    # ============================================================
    # 2) Noise fraction plots — (coherent / incoherent noise)
    # ============================================================
    ep.compute_and_plot_coherent_noise(
        "combined", eval_cfg,
        variants, residuals,
        os.path.join(plot_dir, "coherent_noise"),
        trunc_fracs=(1.0, 0.95, 0.90)
    )

    # ============================================================
    # 3) Eigendecomposition (CM EXCLUDED)
    # ============================================================
    ep.plot_all_eigenvalues(
        "combined", variants, residuals,
        os.path.join(plot_dir, "eigenvalues")
    )

    ep.plot_all_eigenvectors(
        eval_cfg, "combined", variants, residuals,
        3, os.path.join(plot_dir, "eigenvectors")
    )

    ep.plot_all_projection_hists(
        "combined", variants, residuals,
        split.cm_df, 3,
        os.path.join(plot_dir, "eigenprojections")
    )

    print(f"[OK] Method B DNN1 plots saved → {plot_dir}")




def run_method_B_model_inference(cfg):
    """
    Extended Method B Inference (multi-model support):
    - If cfg.method_B_layout_type is a string -> run only that model
    - If it's a list -> run all models in the list
    - If it's None -> automatically detect all available models under dnn_models/
    """

    # --- Check if Method B prediction is enabled ---
    if not getattr(cfg, "enable_method_B_prediction", False):
        print("[INFO] Method B model prediction disabled in config.")
        return

    # --- Main configuration parameters ---
    trained_module = cfg.method_B_trained_module
    evaluated_modules = cfg.method_B_eval_modules
    layout_types = cfg.method_B_layout_type  # can be str, list, or None

    # --- Normalize to a list ---
    if layout_types is None:
        layout_types = []
    elif isinstance(layout_types, str):
        layout_types = [layout_types]

    # --- Iterate over evaluated modules ---
    for evaluated_module in evaluated_modules:
        print(f"\n[METHOD B INFERENCE - NEW] Evaluating module: {evaluated_module}")

        # ------------------------------------------------------------------
        # 1. Detect available models if layout_types is empty
        # ------------------------------------------------------------------
        model_root = os.path.join(
            "./method_B",
            f"trained_{trained_module}",
            f"evaluated_{evaluated_module}",
            "dnn",
            "dnn_models"
        )

        if not layout_types:
            try:
                layout_types = sorted([
                    d for d in os.listdir(model_root)
                    if os.path.isdir(os.path.join(model_root, d))
                ])
                print(f"[AUTO] Detected {len(layout_types)} model(s): {layout_types}")
            except FileNotFoundError:
                print(f"[ERROR] No dnn_models directory found: {model_root}")
                continue

        # ------------------------------------------------------------------
        # 2. Loop through all model types
        # ------------------------------------------------------------------
        for model_type in layout_types:
            print(f"\n[INFO] Using model type: {model_type}")

            # --- Define paths ---
            model_dir = os.path.join(model_root, model_type)
            weights_path = os.path.join(model_dir, "regression_dnn_best.pth")

            input_dir = os.path.join(
                "./method_B",
                f"trained_{trained_module}",
                f"evaluated_{evaluated_module}",
                "dnn",
                "dnn_inputs"
            )

            output_dir = os.path.join(
                "./method_B",
                f"trained_{trained_module}",
                f"evaluated_{evaluated_module}",
                "dnn",
                "dnn_outputs2",
                model_type
            )
            os.makedirs(output_dir, exist_ok=True)

            plot_dir = os.path.join(
                "./plots/performance",
                f"trained_{trained_module}",
                f"evaluated_{evaluated_module}",
                "dnn_plots2",
                model_type
            )
            os.makedirs(plot_dir, exist_ok=True)

            # --- Check for weights file ---
            if not os.path.exists(weights_path):
                print(f"[SKIP] Model weights not found: {weights_path}")
                continue

            # --- Load DNN inputs ---
            try:
                inputs_train = np.load(os.path.join(input_dir, "inputs_train.npy"))
                inputs_val = np.load(os.path.join(input_dir, "inputs_val.npy"))
                inputs_combined = np.concatenate([inputs_train, inputs_val], axis=0)
                print(f"[OK] Loaded DNN inputs from {input_dir}, shape={inputs_combined.shape}")
            except Exception as e:
                print(f"[ERROR] Failed to load DNN inputs: {e}")
                continue

            # --- Load true measurement data ---
            true_path = os.path.join("./method_A", evaluated_module, "analytic", "meas_true.pkl")
            if not os.path.exists(true_path):
                print(f"[SKIP] True measurements not found for {evaluated_module}: {true_path}")
                continue

            meas_true = pd.read_pickle(true_path)
            print(f"[OK] Loaded true measurements from {true_path}, shape={meas_true.shape}")

            # --- Build and load DNN model ---
            if model_type.startswith("in") and "__" in model_type:
                try:
                    input_dim = int(model_type.split("__")[0].replace("in", ""))
                except ValueError:
                    input_dim = inputs_combined.shape[1]
            else:
                input_dim = inputs_combined.shape[1]


            # --- Automatically detect layer structure from model name ---
            if model_type.startswith("in") and "__" in model_type:
                try:
                    # Extract architecture section between first and second "__"
                    layer_str = model_type.split("__")[1]
                    layer_sizes = [int(x) for x in layer_str.split("-") if x.isdigit()]
                    print(f"[AUTO] Detected architecture from model name: {layer_sizes}")
                except Exception:
                    layer_sizes = cfg.nodes_per_layer
                    print(f"[WARN] Could not parse architecture from name, using default: {layer_sizes}")
            else:
                layer_sizes = cfg.nodes_per_layer


            #model = DNNFlex(input_dim, cfg.nodes_per_layer, cfg.dropout_rate)
            model = DNNFlex(input_dim, layer_sizes, cfg.dropout_rate)
            inferencer = DNNInferencer(
                model=model,
                weights_path=weights_path,
                dtype=np.float32,
                batch_size=16384
            )

            # --- Run prediction ---

            # --- Run prediction ---
            preds_flat = inferencer(inputs_combined)

            # --- Align predictions with measurement indices (same as Method A) ---
            eval_cfg = ep.EvalConfig(
                modulenames_used_for_training=[evaluated_module],
                modulename_for_evaluation=evaluated_module,
                nodes_per_layer=cfg.nodes_per_layer,
                dropout_rate=cfg.dropout_rate,
                modeltag=cfg.modeltag,
                inputfoldertag="",
                ncmchannels=cfg.ncmchannels,
                nch_per_erx=cfg.nch_per_erx,
            )
            io = ep.DataIO(eval_cfg)
            io.load_all()
            split = io.get_split("combined")

            # 🔧 Convert flat predictions into event-channel table
            preds_df = ep.pivot_flat_preds_to_event_channel(
                preds_flat,
                split.eventid_flat,
                split.channels_flat,
                split.measurements_df
            )


            # --- Compute residuals ---
            if len(meas_true) != len(preds_df):
                min_len = min(len(meas_true), len(preds_df))
                meas_true = meas_true.iloc[:min_len]
                preds_df = preds_df.iloc[:min_len]

            residual_df = meas_true - preds_df

            # --- Save results ---
            meas_true.to_pickle(os.path.join(output_dir, "true.pkl"))
            preds_df.to_pickle(os.path.join(output_dir, "pred.pkl"))
            residual_df.to_pickle(os.path.join(output_dir, "residual.pkl"))
            print(f"[OK] Results saved → {output_dir}")

            # --- Skip plot generation if disabled in config ---
            if not cfg.method_B_enable_plots:
                print(f"[SKIP] Method B plots disabled in config for {evaluated_module}.")
                continue

            # --- Generate performance plots ---
            try:
                eval_cfg = ep.EvalConfig(
                    modulenames_used_for_training=[trained_module],
                    modulename_for_evaluation=evaluated_module,
                    nodes_per_layer=cfg.nodes_per_layer,
                    dropout_rate=cfg.dropout_rate,
                    modeltag=cfg.modeltag,
                    inputfoldertag="",
                    ncmchannels=cfg.ncmchannels,
                    nch_per_erx=cfg.nch_per_erx,
                )
                io = ep.DataIO(eval_cfg)
                io.load_all()
                split = io.get_split("combined")

                variants = {"true": meas_true, "dnn_pred": preds_df}
                residuals = {"dnn_pred": residual_df}

                variants_with_cms = {
                    k: ep.add_cms_to_measurements_df(v, split.cm_df, drop_constant_cm=False)
                    for k, v in variants.items()
                }
                residuals_with_cms = {
                    "dnn_pred": ep.add_cms_to_measurements_df(residual_df, split.cm_df, drop_constant_cm=False)
                }

                ep.plot_cov_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                                 os.path.join(plot_dir, "covcorr"))
                ep.plot_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                                  split.cm_df, os.path.join(plot_dir, "distcorr"))
                ep.plot_delta_lin_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                                            split.cm_df, os.path.join(plot_dir, "delta_lin_dist_corr"))
                ep.plot_all_eigenvalues("combined", variants_with_cms, residuals_with_cms,
                                        os.path.join(plot_dir, "eigenvalues_cmincl"))
                ep.plot_all_eigenvectors(eval_cfg, "combined", variants_with_cms, residuals_with_cms,
                                         3, os.path.join(plot_dir, "eigenvectors_cmincl"))
                ep.plot_all_projection_hists("combined", variants, residuals,
                                             split.cm_df, 3, os.path.join(plot_dir, "eigenprojections"))

                print(f"[OK] Plots saved → {plot_dir}")

            except Exception as e:
                print(f"[WARN] Plot generation failed for {evaluated_module} ({model_type}): {e}")

    print("\n[COMPLETE] Multi-model Method B inference finished successfully!\n")



def run_dnn_block(cfg, module):
    """
    Runs the DNN prediction and plotting steps for a given module,
    independently of analytic computation.
    """
    print(f"\n[DNN] Starting independent DNN block for {module}")

    # --- Step 1: Run DNN prediction ---
    if cfg.run_dnn_predict and (cfg.target_modules is None or module in cfg.target_modules):
        print(f"[INFO] Running DNN prediction for module: {module}")
        run_dnn_prediction_for_module(cfg, module)
    else:
        print(f"[SKIP] DNN prediction disabled or module not selected: {module}")

    # --- Step 2: Plot DNN results ---
    if cfg.enable_dnn_plots and (cfg.target_modules is None or module in cfg.target_modules):
        print(f"[INFO] Generating DNN performance plots for module: {module}")
        try:
            generate_dnn_plots_for_module(cfg, module)
        except Exception as e:
            print(f"[ERROR] DNN plot generation failed for {module}: {e}")
    else:
        print(f"[SKIP] DNN plots disabled or module not selected: {module}")


def generate_dnn_plots_for_module(cfg, module):
    """
    Generate correlation, eigenvalue, and projection plots
    for DNN predictions independently of the analytic block.
    """
    import os
    import pandas as pd
    import evaluate_performance as ep

    print(f"[INFO] Generating independent DNN plots for {module}")

    if isinstance(cfg.dnn_model_subpath, str):
        model_folder_name = cfg.dnn_model_subpath.replace("dnn/dnn_models/", "").replace("/", "_")
    else:
        print("[ERROR] Multiple model paths provided but generate_dnn_plots_for_module supports only one. Fix needed.")
        return

    dnn_output_dir = os.path.join(cfg.base_output_dir, module, "dnn", "dnn_outputs", model_folder_name)

    if isinstance(cfg.dnn_model_subpath, str):
        model_folder_name = cfg.dnn_model_subpath.replace("dnn/dnn_models/", "").replace("/", "_")
    else:
        print("[ERROR] Multiple model paths provided but generate_dnn_plots_for_module supports only one. Fix needed.")
        return

    plot_dir_dnn = os.path.join(cfg.plots_output_dir, module, "dnn_predict", model_folder_name, "dnn")
    os.makedirs(plot_dir_dnn, exist_ok=True)

    meas_true_path = os.path.join(cfg.base_output_dir, module, "analytic", "meas_true.pkl")
    dnn_pred_path = os.path.join(dnn_output_dir, f"predictions_{module}_combined.pkl")

    if not (os.path.exists(meas_true_path) and os.path.exists(dnn_pred_path)):
        print(f"[SKIP] Missing DNN prediction or meas_true for {module}, skipping DNN plots.")
        return

    meas_true = pd.read_pickle(meas_true_path)
    dnn_pred = pd.read_pickle(dnn_pred_path)
    residual_dnn = meas_true - dnn_pred

    eval_cfg = ep.EvalConfig(
        modulenames_used_for_training=[module],
        modulename_for_evaluation=module,
        nodes_per_layer=cfg.nodes_per_layer,
        dropout_rate=cfg.dropout_rate,
        modeltag=cfg.modeltag,
        inputfoldertag="",
        ncmchannels=cfg.ncmchannels,
        nch_per_erx=cfg.nch_per_erx,
    )
    io = ep.DataIO(eval_cfg)
    io.load_all()
    split = io.get_split("combined")

    variants = {"true": meas_true, "dnn": dnn_pred}
    variants_with_cms = {
        k: ep.add_cms_to_measurements_df(v, split.cm_df, drop_constant_cm=False)
        for k, v in variants.items()
    }
    residuals = {"dnn": residual_dnn}
    residuals_with_cms = {
        "dnn": ep.add_cms_to_measurements_df(residual_dnn, split.cm_df, drop_constant_cm=False)
    }

    # === Generate DNN plots ===
    ep.plot_cov_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                     os.path.join(plot_dir_dnn, "covcorr"))
    ep.plot_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                      split.cm_df, os.path.join(plot_dir_dnn, "distcorr"))
    ep.plot_delta_lin_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                                split.cm_df, os.path.join(plot_dir_dnn, "delta_lin_dist_corr"))
    ep.plot_all_eigenvalues("combined", variants_with_cms, residuals_with_cms,
                            os.path.join(plot_dir_dnn, "eigenvalues_cmincl"))
    ep.plot_all_eigenvectors(eval_cfg, "combined", variants_with_cms, residuals_with_cms,
                             3, os.path.join(plot_dir_dnn, "eigenvectors_cmincl"))
    ep.plot_all_projection_hists("combined", variants, residuals,
                                 split.cm_df, 3, os.path.join(plot_dir_dnn, "eigenprojections"))

def run_control_predict_dnn_only(cfg):
    from models import DNNFlex
    from evaluate_performance import pivot_flat_preds_to_event_channel

    if not cfg.enable_control_predict:
        print("[CONTROL] Control-predict pipeline disabled.")
        return

    print("\n================ CONTROL-PREDICT PIPELINE ================\n")

    for layout in cfg.control_model_layouts:
        print(f"\n[CONTROL] Evaluating model layout: {layout}")

        for trained_mod in cfg.control_trained_modules:

            model_dir = os.path.join(cfg.control_model_root, trained_mod, layout)
            weights_path = os.path.join(model_dir, "regression_dnn_best.pth")

            print(f"[CONTROL] Loading weights from: {weights_path}")

            if not os.path.exists(weights_path):
                print(f"[WARNING] Weights not found → skipping layout {layout}")
                continue

            for eval_mod in cfg.control_evaluated_modules:
                print(f"[CONTROL] Predicting on evaluated module {eval_mod}")

                areimers_dir = os.path.join(cfg.control_input_root, eval_mod)

                X_train = np.load(os.path.join(areimers_dir, "inputs_train.npy"))
                X_val   = np.load(os.path.join(areimers_dir, "inputs_val.npy"))
                X_all   = np.concatenate([X_train, X_val], axis=0)

                # Load full split info
                eval_cfg_inputs = ep.EvalConfig(
                    modulenames_used_for_training=[eval_mod],
                    modulename_for_evaluation=eval_mod,
                    nodes_per_layer=[1],
                    dropout_rate=0.0,
                    modeltag="",
                    inputfoldertag="",
                    nch_per_erx=cfg.control_nch_per_erx,
                    ncmchannels=cfg.control_ncmchannels,
                )
                io = ep.DataIO(eval_cfg_inputs)
                io.load_all()
                split = io.get_split("combined")

                # Parse architecture
                parts = layout.split("__")
                arch_str = parts[1]
                nodes = [int(x) for x in arch_str.split("-")]

                model = DNNFlex(X_all.shape[1], nodes, cfg.control_dropout_rate)
                state_dict = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(state_dict)
                model.eval()


                # ======================================================
                # PERFORMANCE METRICS START
                # ======================================================
                n_samples = X_all.shape[0]
                input_dim = X_all.shape[1]
                flops_per_sample = estimate_dense_flops_per_sample(input_dim, nodes, 1)
                total_flops = flops_per_sample * n_samples

                t0_wall = time.perf_counter()
                t0_cpu  = time.process_time()


                # Run prediction with safe batching to avoid OOM
                X_tensor = torch.tensor(X_all, dtype=torch.float32)
                pred_list = []
                batch_size = 20000   # safe batch size for lxplus

                with torch.no_grad():
                    for i in range(0, X_tensor.shape[0], batch_size):
                        batch = X_tensor[i:i+batch_size]
                        pred_batch = model(batch).cpu().numpy()
                        pred_list.append(pred_batch)

                preds = np.concatenate(pred_list, axis=0)

                # ======================================================
                # PERFORMANCE METRICS END
                # ======================================================
                t1_wall = time.perf_counter()
                t1_cpu  = time.process_time()

                wall_time = t1_wall - t0_wall
                cpu_time  = t1_cpu - t0_cpu
                throughput = n_samples / wall_time
                gflops = total_flops / 1e9
                gflops_per_sec = gflops / wall_time

                print(f"[CONTROL] Wall time:  {wall_time:.3f} s")
                print(f"[CONTROL] CPU time:   {cpu_time:.3f} s")
                print(f"[CONTROL] Samples:    {n_samples:,}")
                print(f"[CONTROL] Throughput: {throughput:,.1f} samples/s")
                print(f"[CONTROL] GFLOP/s:    {gflops_per_sec:.2f}")


                print(f"[CONTROL] Prediction done → shape {preds.shape}")


                preds_df = pivot_flat_preds_to_event_channel(
                    preds.flatten(),
                    split.eventid_flat,
                    split.channels_flat,
                    split.measurements_df
                )

                meas_true = split.measurements_df.copy()
                if meas_true.index.nlevels == 1:
                    meas_true_pivot = meas_true
                else:
                    meas_true.index.set_names(["event", "channel"], inplace=True)
                    meas_true_pivot = meas_true.unstack(level="channel")

                residual_pivot = meas_true_pivot - preds_df

                out_dir = os.path.join(cfg.control_output_root, trained_mod, eval_mod, layout)
                os.makedirs(out_dir, exist_ok=True)

                meas_true_pivot.to_pickle(os.path.join(out_dir, "true.pkl"))
                preds_df.to_pickle(os.path.join(out_dir, "pred.pkl"))
                residual_pivot.to_pickle(os.path.join(out_dir, "residual.pkl"))

                # ======================================================
                # SAVE PERFORMANCE METRICS
                # ======================================================
                perf_path = os.path.join(out_dir, "performance_metrics.txt")
                with open(perf_path, "w") as f:
                    f.write(f"layout: {layout}\n")
                    f.write(f"trained_module: {trained_mod}\n")
                    f.write(f"eval_module: {eval_mod}\n")
                    f.write(f"samples: {n_samples}\n")
                    f.write(f"wall_time: {wall_time:.6f}\n")
                    f.write(f"cpu_time: {cpu_time:.6f}\n")
                    f.write(f"gpu_time: -1\n")
                    f.write(f"throughput_samples_per_sec: {throughput:.3f}\n")
                    f.write(f"flops_per_sample: {flops_per_sample}\n")
                    f.write(f"total_flops: {total_flops}\n")
                    f.write(f"gflops_per_sec: {gflops_per_sec:.3f}\n")

                print(f"[CONTROL] Saved performance metrics → {perf_path}")


                print(f"[CONTROL] Saved PKLs → {out_dir}")

                if cfg.enable_control_plots:
                    generate_control_plots(
                        eval_cfg_inputs,
                        meas_true_pivot,
                        preds_df,
                        residual_pivot,
                        split,
                        out_dir
                    )

    print("\n================ CONTROL-PREDICT COMPLETE ================\n")

def generate_control_plots(eval_cfg, meas_true, dnn_pred, residual, split, plot_dir):
    import os
    import evaluate_performance as ep

    variants = {"true": meas_true, "dnn": dnn_pred}
    residuals = {"dnn": residual}

    variants_with_cms = {
        k: ep.add_cms_to_measurements_df(v, split.cm_df, drop_constant_cm=False)
        for k, v in variants.items()
    }

    residuals_with_cms = {
        "dnn": ep.add_cms_to_measurements_df(residual, split.cm_df, drop_constant_cm=False)
    }

    subdirs = [
        "covcorr", "distcorr", "delta_lin_dist_corr",
        "coherent_noise", "eigenvalues", "eigenvectors",
        "eigenvectors/eigenvectors_2d", "eigenprojections"
    ]
    for sd in subdirs:
        os.makedirs(os.path.join(plot_dir, sd), exist_ok=True)

    print(f"[CONTROL-PLOT] Saving plots to: {plot_dir}")

    ep.plot_cov_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                     os.path.join(plot_dir, "covcorr"))

    ep.plot_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                      split.cm_df, os.path.join(plot_dir, "distcorr"))

    ep.plot_delta_lin_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms,
                                split.cm_df, os.path.join(plot_dir, "delta_lin_dist_corr"))

    ep.compute_and_plot_coherent_noise("combined", eval_cfg,
                                       variants, residuals,
                                       os.path.join(plot_dir, "coherent_noise"))

    ep.plot_all_eigenvalues("combined", variants, residuals,
                            os.path.join(plot_dir, "eigenvalues"))

    ep.plot_all_eigenvectors(eval_cfg, "combined", variants, residuals,
                             3, os.path.join(plot_dir, "eigenvectors"))

    ep.plot_all_projection_hists("combined", variants, residuals,
                                 split.cm_df, 3,
                                 os.path.join(plot_dir, "eigenprojections"))

    print("[CONTROL-PLOT] Done.")


# =================BASLANGIC =====================

def estimate_dense_flops_per_sample(input_dim, hidden_nodes, output_dim=1):
    """
    Approx FLOPs for a dense forward pass.
    2 * in_dim * out_dim per layer (mul+add).
    """
    layers = [input_dim] + list(hidden_nodes) + [output_dim]
    flops = 0
    for i in range(len(layers) - 1):
        flops += 2 * layers[i] * layers[i+1]
    return flops


def run_control_predict(cfg):
    from models import DNNFlex
    from evaluate_performance import pivot_flat_preds_to_event_channel

    if not cfg.enable_control_predict_ftuning:
        print("[CONTROL] Control-predict pipeline disabled.")
        return

    print("\n================ CONTROL-PREDICT PIPELINE ================\n")

    for layout in cfg.control_model_layouts_ftuning:
        print(f"\n[CONTROL] Evaluating model layout: {layout}")

        for trained_mod in cfg.control_trained_modules_ftuning:

            # Fine-tuning model path
            model_dir = os.path.join(cfg.control_model_root_ftuning, layout)
            weights_path = os.path.join(model_dir, "regression_dnn_best.pth")

            print(f"[CONTROL] Loading weights from: {weights_path}")

            if not os.path.exists(weights_path):
                print(f"[WARNING] Weights not found → skipping layout {layout}")
                continue

            for eval_mod in cfg.control_evaluated_modules_ftuning:
                print(f"[CONTROL] Predicting on evaluated module {eval_mod}")

                # LOAD INPUTS
                areimers_dir = os.path.join(cfg.control_input_root_ftuning, eval_mod)
                X_train = np.load(os.path.join(areimers_dir, "inputs_train.npy"))
                X_val   = np.load(os.path.join(areimers_dir, "inputs_val.npy"))
                X_all   = np.concatenate([X_train, X_val], axis=0)

                # LOAD SPLIT
                eval_cfg_inputs = ep.EvalConfig(
                    modulenames_used_for_training=[eval_mod],
                    modulename_for_evaluation=eval_mod,
                    nodes_per_layer=[1],
                    dropout_rate=0.0,
                    modeltag="",
                    inputfoldertag="",
                    nch_per_erx=cfg.control_nch_per_erx,
                    ncmchannels=cfg.control_ncmchannels,
                )
                io = ep.DataIO(eval_cfg_inputs)
                io.load_all()
                split = io.get_split("combined")

                # PARSE ARCHITECTURE
                arch_str = layout.split("__")[1]
                nodes = [int(x) for x in arch_str.split("-")]

                # CREATE MODEL
                input_dim = X_all.shape[1]
                model = DNNFlex(input_dim, nodes, cfg.control_dropout_rate)
                state_dict = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(state_dict)
                model.eval()

                # ---------------------------
                #   PERFORMANCE TIMERS START
                # ---------------------------
                n_samples = X_all.shape[0]
                flops_per_sample = estimate_dense_flops_per_sample(input_dim, nodes, 1)
                total_flops = flops_per_sample * n_samples

                t0_wall = time.perf_counter()
                t0_cpu  = time.process_time()

                # ---------------------------
                #   PREDICTION
                # ---------------------------
                X_tensor = torch.tensor(X_all, dtype=torch.float32)
                pred_list = []
                batch_size = 20000

                with torch.no_grad():
                    for i in range(0, X_tensor.shape[0], batch_size):
                        batch = X_tensor[i:i+batch_size]
                        pred_batch = model(batch).cpu().numpy()
                        pred_list.append(pred_batch)

                preds = np.concatenate(pred_list, axis=0)

                # ---------------------------
                #   PERFORMANCE TIMERS END
                # ---------------------------
                t1_wall = time.perf_counter()
                t1_cpu  = time.process_time()

                wall_time = t1_wall - t0_wall
                cpu_time  = t1_cpu - t0_cpu
                throughput = n_samples / wall_time
                gflops = total_flops / 1e9
                gflops_per_sec = gflops / wall_time

                print(f"[CONTROL] Prediction done → shape {preds.shape}")
                print(f"[CONTROL] Wall time:  {wall_time:.3f} s")
                print(f"[CONTROL] CPU time:   {cpu_time:.3f} s")
                print(f"[CONTROL] Samples:    {n_samples:,}")
                print(f"[CONTROL] Throughput: {throughput:,.1f} samples/s")
                print(f"[CONTROL] GFLOP/s:    {gflops_per_sec:.2f}")

                # ---------------------------
                #   SAVE PRED / TRUE / RESIDUAL
                # ---------------------------
                out_dir = os.path.join(cfg.control_output_root, layout)
                os.makedirs(out_dir, exist_ok=True)

                preds_df = pivot_flat_preds_to_event_channel(
                    preds.flatten(),
                    split.eventid_flat,
                    split.channels_flat,
                    split.measurements_df
                )

                meas_true = split.measurements_df.copy()
                if meas_true.index.nlevels == 1:
                    meas_true_pivot = meas_true
                else:
                    meas_true.index.set_names(["event", "channel"], inplace=True)
                    meas_true_pivot = meas_true.unstack(level="channel")

                residual_pivot = meas_true_pivot - preds_df

                meas_true_pivot.to_pickle(os.path.join(out_dir, "true.pkl"))
                preds_df.to_pickle(os.path.join(out_dir, "pred.pkl"))
                residual_pivot.to_pickle(os.path.join(out_dir, "residual.pkl"))

                # ---------------------------
                #   SAVE PERFORMANCE METRICS
                # ---------------------------
                perf_path = os.path.join(out_dir, "performance_metrics.txt")
                with open(perf_path, "w") as f:
                    f.write(f"layout: {layout}\n")
                    f.write(f"eval_module: {eval_mod}\n")
                    f.write(f"samples: {n_samples}\n")
                    f.write(f"wall_time: {wall_time:.6f}\n")
                    f.write(f"cpu_time: {cpu_time:.6f}\n")
                    f.write(f"gpu_time: -1\n")  # GPU disabled
                    f.write(f"throughput_samples_per_sec: {throughput:.3f}\n")
                    f.write(f"flops_per_sample: {flops_per_sample}\n")
                    f.write(f"total_flops: {total_flops}\n")
                    f.write(f"gflops_per_sec: {gflops_per_sec:.3f}\n")

                print(f"[CONTROL] Saved performance metrics → {perf_path}")

                # ---------------------------
                #   PLOTS
                # ---------------------------
                if cfg.enable_control_plots_ftuning:
                    plot_dir = os.path.join(cfg.control_plots_root_ftuning,
                                            f"evaluated_{eval_mod}",
                                            layout)
                    os.makedirs(plot_dir, exist_ok=True)

                    generate_control_plots(
                        eval_cfg_inputs,
                        meas_true_pivot,
                        preds_df,
                        residual_pivot,
                        split,
                        plot_dir
                    )

    print("\n================ CONTROL-PREDICT COMPLETE ================\n")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    cfg = CompareConfig()

    # --- Step 1: Analytic pipeline (Method A) ---
    compare_methods(cfg)

    # --- Step 4: DNN Prediction + Plots (always run if enabled) ---
    if cfg.run_dnn_predict:
        for module in cfg.target_modules:
            print(f"\n[DNN] Starting full DNN block for {module} (prediction + plots)...")
            run_dnn_block(cfg, module)

    # --- Step 2: Combined Analytic + DNN Plots ---
    if cfg.enable_combined_dnn_analytic_plots:
        print("\n[STEP] Generating combined analytic + DNN comparison plots...")
        plot_combined_analytic_dnn(cfg)


    # --- Step 2.1: Method A results.txt → comparison charts (4 plots total)
    if cfg.enable_methodA_results_comparison:
        generate_methodA_comparison_plots(cfg)

    compute_coherent_noise_methodA(cfg)


    # --- Step 3: Method B ---
    if cfg.enable_method_B:
        run_method_B(cfg)

    if cfg.enable_method_B_prediction:
        run_method_B_model_inference(cfg)
    """

    if cfg.enable_method_B:
        run_method_B_as_method_A(cfg)

    """
    # --- Step 5: Combined Method B DNN Plots ---
    if cfg.enable_combined_method_B_plots:
        print("\n[STEP] Generating combined Method B DNN plots...")
        plot_combined_method_B_dnn(cfg)


    compute_coherent_noise_methodA(cfg)

    if cfg.enable_control_predict_ftuning:
        run_control_predict(cfg)

    if cfg.enable_control_predict:
        run_control_predict_dnn_only(cfg)

