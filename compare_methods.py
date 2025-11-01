import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
import evaluate_performance as ep  # Import analytic & plotting utilities


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

    enable_plots: bool = True
    skip_existing_data: bool = True     # Skip analytic data if residual.pkl exists
    skip_existing_plots: bool = True    # Skip plot generation if plot folder exists
    drop_constant_cm: bool = True
    verbose: bool = True

    nodes_per_layer: list = None
    dropout_rate: float = 0.0
    modeltag: str = ""
    ncmchannels: int = 12
    nch_per_erx: int = 37

    def __post_init__(self):
        if self.nodes_per_layer is None:
            self.nodes_per_layer = [512, 512, 512, 512, 64]


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
            print(f"[SKIP] Analytic data already exist for {module}, skipping computation.")
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

            analytic.fit(split)
            analytic_pred = analytic.predict(split_predict=split, split_correction=split)
            meas_true = split.measurements_df

            assert np.array_equal(meas_true.index, analytic_pred.index)


            residual = pd.DataFrame({
                "adc": (meas_true - analytic_pred).mean(axis=1).to_numpy()
            })

            # --- Step 7: Save analytic outputs
            meas_true.to_pickle(os.path.join(analytic_output_dir, "meas_true.pkl"))
            analytic_pred.to_pickle(os.path.join(analytic_output_dir, "analytic_pred.pkl"))
            residual.to_pickle(os.path.join(analytic_output_dir, "residual.pkl"))
            print(f"[OK] Analytic results saved → {analytic_output_dir}")

            # --- Step 8: Export analytic outputs for DNN (train/val)
            n = len(analytic_pred)
            n_train = int(0.8 * n)

            # --- Prepare Areimers-style DNN inputs (use ~20 original features) ---
            with open(os.path.join(module_dir, "colnames.json"), "r") as f:
                canonical_colnames = json.load(f)

            inputs_df = split.inputs_df.copy()
            kept_cols = [c for c in canonical_colnames if c in inputs_df.columns and c != "event_id"]
            dnn_inputs = inputs_df[kept_cols].to_numpy()

            # --- Export for DNN ---
            n = len(dnn_inputs)
            n_train = int(0.8 * n)

            np.save(os.path.join(dnn_output_dir, "inputs_train.npy"), dnn_inputs[:n_train])
            np.save(os.path.join(dnn_output_dir, "inputs_val.npy"), dnn_inputs[n_train:])
            np.save(os.path.join(dnn_output_dir, "targets_train.npy"), residual.iloc[:n_train].to_numpy())
            np.save(os.path.join(dnn_output_dir, "targets_val.npy"), residual.iloc[n_train:].to_numpy())

            # --- Save matching colnames.json ---
            with open(os.path.join(dnn_output_dir, "colnames.json"), "w") as f:
                json.dump(kept_cols, f)

            print(f"[OK] Exported DNN-ready inputs (~{len(kept_cols)}) and targets → {dnn_output_dir}")



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

    print("\n[COMPLETE] compare_methods pipeline finished successfully!\n")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    cfg = CompareConfig(
        base_input_dir="/eos/user/a/areimers/hgcal/dnn_inputs",
        base_output_dir="./method_A",
        reference_module="ML_F3W_WXIH0191",
        enable_plots=True,
        skip_existing_data=False,    # Skip analytic computation if data already exist
        skip_existing_plots=True,  # Always re-generate plots
        drop_constant_cm=True,
        verbose=True,
        nodes_per_layer=[512, 512, 512, 512, 64],
        dropout_rate=0.0
    )

    compare_methods(cfg)
