import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
import evaluate_performance as ep  # Import analytic & plotting utilities
from dataclasses import dataclass, field
import gzip
import pickle
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

    #Analytic Calculations with dnn_inputs
    enable_plots: bool = True
    skip_existing_data: bool = True     # Skip analytic data if residual.pkl exists
    skip_existing_plots: bool = True    # Skip plot generation if plot folder exists
    drop_constant_cm: bool = True
    verbose: bool = True

    # --- DNN configuration ---
    run_dnn_predict: bool = False
    enable_dnn_plots: bool = False
    dnn_model_subpath: str = "dnn/dnn_models/in20__512-512-512-512-64__dr0"
    dnn_weights_name: str = "regression_dnn_best.pth"
    target_modules: list[str] = field(default_factory=lambda: ["ML_F3W_WXIH0191"])

    nodes_per_layer: list = None
    dropout_rate: float = 0.0
    modeltag: str = ""
    ncmchannels: int = 12
    nch_per_erx: int = 37

    # --- Analytic residuals configuration ---
    enable_analytic_residuals: bool = False
    analytic_residual_modules: list[str] = field(default_factory=lambda: ["ML_F3W_WXIH0191"])
    analytic_residual_input_root: str = "/eos/user/g/gmihriye/CM/compare_methods/CMCorrection/analytic_residuals/inputs"
    analytic_residual_output_root: str = "/eos/user/g/gmihriye/CM/compare_methods/CMCorrection/analytic_residuals/outputs"
    analytic_residual_plots_root: str = "./plots/performance"


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


def run_dnn_prediction_for_module(cfg, module):
    """
    Run DNN prediction for a given module, align with analytic data,
    and save both raw and CM-augmented outputs for plotting.
    """
    import torch
    from models import DNNFlex
    from evaluate_performance import DNNInferencer, pivot_flat_preds_to_event_channel

    print(f"\n[DNN] Running DNN prediction for module: {module}")

    model_dir = os.path.join(cfg.base_output_dir, module, cfg.dnn_model_subpath)
    weights_path = os.path.join(model_dir, cfg.dnn_weights_name)
    dnn_output_dir = os.path.join(cfg.base_output_dir, module, "dnn", "dnn_outputs")
    os.makedirs(dnn_output_dir, exist_ok=True)

    if not os.path.exists(weights_path):
        print(f"[SKIP] No DNN weights found for {module}: {weights_path}")
        return

    eos_dir = os.path.join(cfg.base_input_dir, module)
    inputs_train = np.load(os.path.join(eos_dir, "inputs_train.npy"))
    inputs_val = np.load(os.path.join(eos_dir, "inputs_val.npy"))
    inputs_combined = np.concatenate([inputs_train, inputs_val], axis=0)

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

    # --- Ensure cm_df has the same index and shape as measurements_df (same as evaluate_performance)
    if "eventid" in split.cm_df.columns:
        split.cm_df = split.cm_df.groupby("eventid").first()

    split.cm_df.index = split.measurements_df.index
    split.cm_df = split.cm_df.loc[:, ~split.cm_df.columns.duplicated()]
    split.cm_df = split.cm_df.reindex(sorted(split.cm_df.columns), axis=1)



    input_dim = inputs_combined.shape[1]
    model = DNNFlex(input_dim, cfg.nodes_per_layer, cfg.dropout_rate)
    inferencer = ep.DNNInferencer(model=model, weights_path=weights_path, dtype=np.float32, batch_size=16384)


    preds_flat = inferencer(inputs_combined)
    preds_df = pivot_flat_preds_to_event_channel(preds_flat, split.eventid_flat, split.channels_flat, split.measurements_df)

    np.save(os.path.join(dnn_output_dir, f"predictions_{module}_combined.npy"), preds_flat)
    preds_df.to_pickle(os.path.join(dnn_output_dir, f"predictions_{module}_combined.pkl"))

    preds_df_with_cm = ep.add_cms_to_measurements_df(preds_df, split.cm_df, drop_constant_cm=True)
    preds_df_with_cm.to_pickle(os.path.join(dnn_output_dir, f"predictions_{module}_withCM_combined.pkl"))
    print(f"[OK] DNN predictions (with CM) saved → {dnn_output_dir}")

    meas_true_path = os.path.join(cfg.base_output_dir, module, "analytic", "meas_true.pkl")
    if not os.path.exists(meas_true_path):
        print(f"[WARN] meas_true.pkl missing for {module}, skipping residual computation.")
        return

    meas_true = pd.read_pickle(meas_true_path)
    meas_true = meas_true.loc[preds_df.index]

    residual_dnn = meas_true - preds_df
    residual_dnn_with_cm = ep.add_cms_to_measurements_df(residual_dnn, split.cm_df, drop_constant_cm=True)

    residual_dnn.to_pickle(os.path.join(dnn_output_dir, f"residual_dnn_{module}.pkl"))
    residual_dnn_with_cm.to_pickle(os.path.join(dnn_output_dir, f"residual_dnn_{module}_withCM.pkl"))
    print(f"[OK] DNN residuals (with CM) saved → {dnn_output_dir}")


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

        # --- Define eval_cfg always (needed for plots even if analytic skipped)
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


        """
        if cfg.skip_existing_data and data_exists:
            print(f"[SKIP] Analytic data already exist for {module}, skipping computation.")
            #continue
        """
        if cfg.skip_existing_data and data_exists:
            print(f"[SKIP] Analytic data already exist for {module}, reloading saved data.")
            analytic_output_dir = os.path.join(cfg.base_output_dir, module, "analytic")

            # Load previously saved analytic results so variables exist
            # Load previously saved analytic results so variables exist
            try:
                meas_true = pd.read_pickle(os.path.join(analytic_output_dir, "meas_true.pkl"))
                analytic_pred = pd.read_pickle(os.path.join(analytic_output_dir, "analytic_pred.pkl"))
                residual = pd.read_pickle(os.path.join(analytic_output_dir, "residual.pkl"))
            except Exception as e:
                raise RuntimeError(f"[ERROR] Could not reload analytic data for {module}: {e}")

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
            analytic_pred = analytic.predict_k(split_predict=split, split_correction=split, k=0)

            meas_true = split.measurements_df
            assert np.array_equal(meas_true.index, analytic_pred.index)

            # Residual = true - analytic_pred (exactly as evaluate_performance)
            residual_df = meas_true - analytic_pred

            # Save in same structure (no stack/reset_index difference)
            residual = residual_df
            # --- Step 7: Save analytic outputs
            """
            meas_true.to_pickle(os.path.join(analytic_output_dir, "meas_true.pkl"))
            analytic_pred.to_pickle(os.path.join(analytic_output_dir, "analytic_pred.pkl"))
            residual.to_pickle(os.path.join(analytic_output_dir, "residual.pkl"))
            """
            meas_true.to_pickle(os.path.join(analytic_output_dir, "meas_true.pkl.gz"), compression="gzip")
            analytic_pred.to_pickle(os.path.join(analytic_output_dir, "analytic_pred.pkl.gz"), compression="gzip")
            residual.to_pickle(os.path.join(analytic_output_dir, "residual.pkl.gz"), compression="gzip")

            print(f"[OK] Analytic results saved → {analytic_output_dir}")
            """
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
            """
        # --- Step 8: Prepare DNN-ready data (copy Areimers inputs + create analytic residual targets)

        print(f"[INFO] Preparing DNN-ready residual targets for {module}...")

        # ① Copy original DNN input files
        original_input_dir = os.path.join(cfg.base_input_dir, module)
        dnn_output_dir = os.path.join(cfg.base_output_dir, module, "dnn", "dnn_inputs")
        os.makedirs(dnn_output_dir, exist_ok=True)

        inputs_train_path  = os.path.join(original_input_dir, "inputs_train.npy")
        inputs_val_path    = os.path.join(original_input_dir, "inputs_val.npy")
        indices_train_path = os.path.join(original_input_dir, "indices_train.npy")
        indices_val_path   = os.path.join(original_input_dir, "indices_val.npy")

        required_files = [inputs_train_path, inputs_val_path, indices_train_path, indices_val_path]
        if not all(os.path.exists(p) for p in required_files):
            raise FileNotFoundError(f"[ERROR] Missing Areimers input files for {module}!")

        import shutil
        for fname in ["inputs_train.npy", "inputs_val.npy", "indices_train.npy", "indices_val.npy", "colnames.json"]:
            src = os.path.join(original_input_dir, fname)
            dst = os.path.join(dnn_output_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"[COPY] {fname} copied from dnn inputs folder.")
            else:
                print(f"[WARN] Missing file: {src}")

        # ② Compute analytic residuals and flatten
        residual_df = meas_true - analytic_pred
        residual_flat = residual_df.to_numpy(dtype=np.float32).reshape(-1, 1, order="C")

        # ③ Split residuals into train and validation sets using Areimers' indices
        idx_train = np.load(indices_train_path)
        idx_val   = np.load(indices_val_path)

        targets_train = residual_flat[idx_train]
        targets_val   = residual_flat[idx_val]

        # ④ Save the new analytic residual targets
        np.save(os.path.join(dnn_output_dir, "targets_train.npy"), targets_train)
        np.save(os.path.join(dnn_output_dir, "targets_val.npy"), targets_val)

        print(f"[OK] Analytic residual targets saved → {dnn_output_dir}")
        print(f"     → targets_train: {targets_train.shape}, targets_val: {targets_val.shape}")


        # --- Run DNN prediction only for selected modules ---
        if cfg.run_dnn_predict:
            if cfg.target_modules is None or module in cfg.target_modules:
                run_dnn_prediction_for_module(cfg, module)
            else:
                print(f"[SKIP] DNN prediction skipped for {module} (not in target list).")



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

                # --- Additional: Noise fraction ratio plot for analytic ---
                noise_plot_dir = os.path.join(plot_dir, "noise_fraction_ratio")
                os.makedirs(noise_plot_dir, exist_ok=True)

                variants_for_noise = {
                    "true": split.measurements_df
                }
                residuals_for_noise = {
                    "analytic": residual
                }

                ep.compute_and_plot_coherent_noise(
                    split_name="combined",
                    cfg=eval_cfg,
                    variants=variants_for_noise,
                    residuals=residuals_for_noise,
                    plot_dir=noise_plot_dir,
                    trunc_fracs=(1.0,)
                )

                print(f"[OK] Analytic noise fraction ratio saved → {noise_plot_dir}")

        # --- Step 10: DNN plots ---
        if cfg.enable_dnn_plots and (cfg.target_modules is None or module in cfg.target_modules):
            print(f"[PLOTS] Generating DNN plots for {module}...")
            dnn_output_dir = os.path.join(cfg.base_output_dir, module, "dnn", "dnn_outputs")
            plot_dir_dnn = os.path.join(cfg.plots_output_dir, module, "dnn")
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

            # --- Additional: Noise fraction ratio plot for DNN ---
            noise_plot_dir_dnn = os.path.join(plot_dir_dnn, "noise_fraction_ratio")
            os.makedirs(noise_plot_dir_dnn, exist_ok=True)

            variants_for_noise = {
                "true": split.measurements_df
            }
            residuals_for_noise = {
                "dnn": residual_dnn
            }

            ep.compute_and_plot_coherent_noise(
                split_name="combined",
                cfg=eval_cfg,
                variants=variants_for_noise,
                residuals=residuals_for_noise,
                plot_dir=noise_plot_dir_dnn,
                trunc_fracs=(1.0,)
            )

            print(f"[OK] DNN noise fraction ratio saved → {noise_plot_dir_dnn}")


        # --- Step 11: Analytic residuals processing ---
        if cfg.enable_analytic_residuals and (cfg.analytic_residual_modules is None or module in cfg.analytic_residual_modules):
            run_analytic_residuals(cfg, module)



    print("\n[COMPLETE] compare_methods pipeline finished successfully!\n")


def run_analytic_residuals(cfg, module):
    """
    Perform analytic inference on residuals obtained from DNN predictions.
    Input residuals are read from dnn_outputs, processed analytically,
    and results + plots are stored in configured output directories.
    """

    print(f"\n[ANALYTIC-RESIDUAL] Processing module: {module}")

    # Define paths
    input_dir = os.path.join(cfg.analytic_residual_input_root, module)
    os.makedirs(input_dir, exist_ok=True)

    dnn_residual_path = os.path.join(cfg.base_output_dir, module, "dnn", "dnn_outputs", f"residual_dnn_{module}.pkl")
    if not os.path.exists(dnn_residual_path):
        print(f"[SKIP] Missing DNN residual for {module}, skipping analytic residual computation.")
        return

    residual_df = pd.read_pickle(dnn_residual_path)

    # Load CM data from analytic base
    analytic_dir = os.path.join(cfg.base_output_dir, module, "analytic")
    cm_ref_path = os.path.join(analytic_dir, "meas_true.pkl")  # reuse structure to get cm_df
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

    # Fit analytic model directly on residuals
    analytic = ep.AnalyticInferencer(drop_constant_cm=cfg.drop_constant_cm)
    analytic.fit(split)
    analytic_pred_residual = analytic.predict(split_predict=split, split_correction=split)

    # Compute corrected residuals
    residual_corrected = residual_df - analytic_pred_residual

    # Save outputs
    output_dir = os.path.join(cfg.base_output_dir, module, "analytic_residual")
    os.makedirs(output_dir, exist_ok=True)
    residual_df.to_pickle(os.path.join(output_dir, f"residual_input_{module}.pkl"))
    analytic_pred_residual.to_pickle(os.path.join(output_dir, f"analytic_pred_residual_{module}.pkl"))
    residual_corrected.to_pickle(os.path.join(output_dir, f"residual_corrected_{module}.pkl"))

    print(f"[OK] Analytic residual outputs saved → {output_dir}")

    # --- Generate plots ---
    plot_dir = os.path.join(cfg.analytic_residual_plots_root, module, "analytic_residuals")
    os.makedirs(plot_dir, exist_ok=True)

    variants = {
        "true": residual_df,  # required by evaluate_performance.plot_all_eigenvectors()
        "residual": residual_df,
        "analytic_residual": analytic_pred_residual
    }


    variants_with_cms = {
        k: ep.add_cms_to_measurements_df(v, split.cm_df, drop_constant_cm=False)
        for k, v in variants.items()
    }
    residuals_with_cms = {
        "analytic_residual": ep.add_cms_to_measurements_df(residual_corrected, split.cm_df, drop_constant_cm=False)
    }


    ep.plot_cov_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms, os.path.join(plot_dir, "covcorr"))
    ep.plot_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms, split.cm_df, os.path.join(plot_dir, "distcorr"))
    ep.plot_delta_lin_dist_corr("combined", eval_cfg, variants_with_cms, residuals_with_cms, split.cm_df, os.path.join(plot_dir, "delta_lin_dist_corr"))
    ep.plot_all_eigenvalues("combined", variants_with_cms, residuals_with_cms, os.path.join(plot_dir, "eigenvalues_cmincl"))
    ep.plot_all_eigenvectors(eval_cfg, "combined", variants_with_cms, residuals_with_cms, 3, os.path.join(plot_dir, "eigenvectors_cmincl"))
    ep.plot_all_projection_hists("combined", variants, residuals, split.cm_df, 3, os.path.join(plot_dir, "eigenprojections"))

    print(f"[OK] Analytic residual plots saved under {plot_dir}\n")

    # --- Additional: Noise fraction ratio plot for analytic_residual ---
    noise_plot_dir_ar = os.path.join(plot_dir, "noise_fraction_ratio")
    os.makedirs(noise_plot_dir_ar, exist_ok=True)

    # true = original DNN residuals, residual = analytically corrected residuals
    dnn_residual_path = os.path.join(
        cfg.base_output_dir, module, "dnn", "dnn_outputs", f"residual_dnn_{module}.pkl"
    )
    if os.path.exists(dnn_residual_path):
        residual_dnn = pd.read_pickle(dnn_residual_path)
        variants_for_noise = {"true": residual_dnn}
    else:
        variants_for_noise = {"true": split.measurements_df}

    residuals_for_noise = {"analytic_residual": residual_corrected}

    ep.compute_and_plot_coherent_noise(
        split_name="combined",
        cfg=eval_cfg,
        variants=variants_for_noise,
        residuals=residuals_for_noise,
        plot_dir=noise_plot_dir_ar,
        trunc_fracs=(1.0,)
    )

    print(f"[OK] Analytic residual noise fraction ratio saved → {noise_plot_dir_ar}")

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    cfg = CompareConfig()
    compare_methods(cfg)
