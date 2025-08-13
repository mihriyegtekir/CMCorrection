import os
import re
import plot_performance
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib as mpl
from collections import defaultdict

# === Optional: Enable dual skip check (also skip if in results.txt or folder exist) ===
ENABLE_RESULTS_TXT_SKIP = True
ENABLE_FOLDER_SKIP = False
existing_result_keys = set()

# Store per-layer fractional RMS improvements
layer_results = defaultdict(list)
# Store detailed dropout-wise fractional RMS improvements
detailed_results = defaultdict(list)

dropout_group_results = defaultdict(list)
dropout_coherent_group_results = defaultdict(list)
coherent_results = defaultdict(list)
coherent_detailed_results = defaultdict(list)

per_module_fractional_means = defaultdict(list)
per_module_coherent_means = defaultdict(list)

RESULTS_FILE = "plots/performance/results.txt"

# Save computed metrics for each evaluated model into a results file.
# Avoids duplication by checking existing entries.

def save_result_to_txt(test_module, train_module, nodes_per_layer, dropout_rate, frac_impr_mean, coh_ratio_mean):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    entry = f"{test_module},{train_module},{'-'.join(map(str, nodes_per_layer))},{dropout_rate},{frac_impr_mean},{coh_ratio_mean}\n"
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            if entry.strip() in (line.strip() for line in f):
                print(f"Duplicate prevented: {test_module}/{train_module}/{nodes_per_layer}/dr{dropout_rate}")
                return
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{test_module},{train_module},{'-'.join(map(str, nodes_per_layer))},{dropout_rate},{frac_impr_mean},{coh_ratio_mean}\n")

# Load previously saved results from RESULTS_FILE
# Rebuilds the global dictionaries for aggregation and plotting

def register_result(test_module, train_module, nodes_per_layer, dropout_rate,
                    frac_impr_mean, coh_ratio_mean):
    save_result_to_txt(test_module, train_module, nodes_per_layer, dropout_rate,
                       frac_impr_mean, coh_ratio_mean)

    per_module_fractional_means[test_module].append(frac_impr_mean)
    per_module_coherent_means[test_module].append(coh_ratio_mean)

    layer_key = tuple(nodes_per_layer)
    layer_results[(test_module, train_module, layer_key)].append(frac_impr_mean)
    coherent_results[(test_module, train_module, layer_key)].append(coh_ratio_mean)

    detailed_results[(test_module, train_module, layer_key)].append(
        (dropout_rate, frac_impr_mean)
    )
    coherent_detailed_results[(test_module, train_module, layer_key)].append(
        (dropout_rate, coh_ratio_mean)
    )

    group_type = "SELF" if is_self(test_module, train_module) else "CROSS"
    dropout_group_results[(group_type, dropout_rate)].append(frac_impr_mean)
    dropout_coherent_group_results[(group_type, dropout_rate)].append(coh_ratio_mean)


def load_existing_results():
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

                group_type = "SELF" if is_self(test_module, train_module) else "CROSS"
                dropout_group_results[(group_type, dropout_val)].append(frac_impr_mean)
                dropout_coherent_group_results[(group_type, dropout_val)].append(coh_ratio_mean)

                layer_results[(test_module, train_module, layer_tuple)].append(frac_impr_mean)
                coherent_results[(test_module, train_module, layer_tuple)].append(coh_ratio_mean)

                per_module_fractional_means[test_module].append(frac_impr_mean)
                per_module_coherent_means[test_module].append(coh_ratio_mean)

                if ENABLE_RESULTS_TXT_SKIP:
                    existing_result_keys.add((test_module, train_module, layer_tuple, dropout_val))
            except ValueError:
                continue


# Discover modules and models
def discover_modules_and_models(username_load_model_from):
    input_base = f"/eos/user/{os.getenv('USER')[0]}/{os.getenv('USER')}/hgcal/dnn_inputs"
    modules = [d for d in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, d))]

    models_base = f"/eos/user/{username_load_model_from[0]}/{username_load_model_from}/hgcal/dnn_models"
    models = {}
    for module in os.listdir(models_base):
        module_path = os.path.join(models_base, module)
        if not os.path.isdir(module_path):
            continue
        model_variants = [d for d in os.listdir(module_path) if os.path.isdir(os.path.join(module_path, d))]
        models[module] = model_variants
    return modules, models

# Skip already processed plots
def filter_existing_plots(modules, models, output_base="plots/performance"):
    filtered = {}
    for test_module in modules:
        models_to_run = []
        for train_module, model_list in models.items():
            for model_name in model_list:
                nodes_per_layer, dropout_rate = parse_model_config(model_name)
                if nodes_per_layer is None or dropout_rate is None:
                    continue

                key = (test_module, train_module, tuple(nodes_per_layer), dropout_rate)
                plot_path = os.path.join(output_base, test_module, train_module, model_name)
                has_results_entry = key in existing_result_keys if ENABLE_RESULTS_TXT_SKIP else False
                has_folder = os.path.exists(plot_path) if ENABLE_FOLDER_SKIP else False

                skip = False
                if ENABLE_RESULTS_TXT_SKIP and ENABLE_FOLDER_SKIP:
                    if has_results_entry and has_folder:
                        skip = True
                        print(f"✅ Skipping {test_module}/{train_module}/{model_name} "
                              f"(results.txt AND folder present)")
                elif ENABLE_RESULTS_TXT_SKIP and not ENABLE_FOLDER_SKIP:
                    if has_results_entry:
                        skip = True
                        print(f"✅ Skipping {test_module}/{train_module}/{model_name} "
                              f"(exists in results.txt)")
                elif ENABLE_FOLDER_SKIP and not ENABLE_RESULTS_TXT_SKIP:
                    if has_folder:
                        skip = True
                        print(f"✅ Skipping {test_module}/{train_module}/{model_name} "
                              f"(folder present)")

                if skip:
                    continue

                models_to_run.append((train_module, model_name))

        if models_to_run:
            filtered[test_module] = models_to_run
    return filtered


# Extracts model architecture (nodes per layer) and dropout rate from the trained model's folder name.
def parse_model_config(model_name):
    nodes = re.search(r"__(\d+(?:-\d+)+)__", model_name)
    if not nodes:
        return None, None
    nodes_per_layer = [int(n) for n in nodes.group(1).split("-")]

    dr = re.search(r"__dr([0-9.]+)", model_name)
    if not dr:
        return None, None
    dropout_rate = float(dr.group(1))
    return nodes_per_layer, dropout_rate


# Determines if the training and evaluation modules are the same (SELF) or different (CROSS) for grouping results.

_MODULE_RE = re.compile(r"ML_[A-Z0-9_]+?(?=_ML_|$)")

def is_self(test_module: str, train_module: str) -> bool:
    train_parts = _MODULE_RE.findall(train_module)
    return test_module in train_parts

def plot_and_save_graphs():
    # Update plot style for professional visualization
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

    orange_color = "#FF7F0E"
    blue_color = "#1F77B4"

    os.makedirs("plots/performance", exist_ok=True)

    # Helper function: Aggregate results across all layers
    # Each model is counted once, grouped by SELF/CROSS and dropout or train module count
    def aggregate_results(data_dict):
       final = defaultdict(list)
       for (group_type, metric), values in data_dict.items():
          mean_val = np.mean(values)
          final[group_type].append((metric, mean_val))
       return final

    # Plot 1: Dropout-Based Mean Fractional RMS Improvement
    dropout_frac = aggregate_results(dropout_group_results)
    train_frac = aggregate_results(trained_group_results)

    plt.figure(figsize=(8, 6), dpi=300)
    for group_type, results in dropout_frac.items():
        results_sorted = sorted(results)
        x = [r[0] for r in results_sorted]
        y = [r[1] for r in results_sorted]
        plt.plot(x, y, '-D', color=blue_color if group_type == "SELF" else orange_color,
                 linewidth=2.2, markersize=8, label=f'{group_type}')
        for xi, yi in zip(x, y):
            plt.text(xi, yi + 0.01, f"{yi:.2f}", fontsize=10, ha='center')
    plt.xlabel('Dropout Rate')
    plt.ylabel('Mean Fractional RMS Improvement')
    plt.title('Dropout-Based Mean Fractional RMS Improvement', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots/performance/dropout_fractional_rms_improvement.pdf")
    plt.close()

    # Plot 2: Dropout-Based Mean Coherent Noise Ratio
    dropout_coh = aggregate_results(dropout_coherent_group_results)
    plt.figure(figsize=(8, 6), dpi=300)
    for group_type, results in dropout_coh.items():
        results_sorted = sorted(results)
        x = [r[0] for r in results_sorted]
        y = [r[1] for r in results_sorted]
        plt.plot(x, y, '-D', color=blue_color if group_type == "SELF" else orange_color,
                 linewidth=2.2, markersize=8, label=f'{group_type}')
        for xi, yi in zip(x, y):
            plt.text(xi, yi + 0.03, f"{yi:.2f}", fontsize=10, ha='center')
    plt.xlabel('Dropout Rate')
    plt.ylabel('Mean Coherent Noise Ratio (corr/uncorr)')
    plt.title('Dropout-Based Mean Coherent Noise Ratio', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots/performance/dropout_coherent_noise_ratio.pdf")
    plt.close()

    # Plot 3: Train Module Count-Based Mean Fractional RMS Improvement
    train_frac  = aggregate_results(trained_group_results)
    plt.figure(figsize=(8, 6), dpi=300)
    for group_type, results in train_frac.items():
        results_sorted = sorted(results)
        x = [int(r[0]) for r in results_sorted]
        y = [r[1] for r in results_sorted]
        plt.plot(x, y, '-D', color=blue_color if group_type == "SELF" else orange_color,
                 linewidth=2.2, markersize=8, label=f'{group_type}')
        for xi, yi in zip(x, y):
            plt.text(xi, yi + 0.01, f"{yi:.2f}", fontsize=10, ha='center')
    plt.xlabel('Number of Train Modules')
    plt.ylabel('Mean Fractional RMS Improvement')
    plt.title('Train Module Count-Based Mean Fractional RMS Improvement', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(sorted(set([int(r[0]) for results in train_frac.values() for r in results])))
    plt.tight_layout()
    plt.savefig("plots/performance/train_module_fractional_rms.pdf")
    plt.close()

    # Plot 4: Train Module Count-Based Mean Coherent Noise Ratio
    train_coh   = aggregate_results(trained_coherent_group_results)
    plt.figure(figsize=(8, 6), dpi=300)
    for group_type, results in train_coh.items():
        results_sorted = sorted(results)
        x = [int(r[0]) for r in results_sorted]
        y = [r[1] for r in results_sorted]
        plt.plot(x, y, '-D', color=blue_color if group_type == "SELF" else orange_color,
                 linewidth=2.2, markersize=8, label=f'{group_type}')
        for xi, yi in zip(x, y):
            plt.text(xi, yi + 0.03, f"{yi:.2f}", fontsize=10, ha='center')
    plt.xlabel('Number of Train Modules')
    plt.ylabel('Mean Coherent Noise Ratio')
    plt.title('Train Module Count-Based Mean Coherent Noise Ratio', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xticks(sorted(set([int(r[0]) for results in train_coh.values() for r in results])))
    plt.tight_layout()
    plt.savefig("plots/performance/train_module_coherent_noise.pdf")
    plt.close()

# === Main Execution Flow ===
# 1. Load existing results
# 2. Discover modules and trained models
# 3. Filter already processed results
# 4. Loop through each (test_module, train_module, model) combination
# 5. Run evaluation, save metrics, and update global dictionaries
# 6. Print summaries and generate plots

def plot_module_mean_histograms():
    import matplotlib.pyplot as plt
    os.makedirs("plots/performance", exist_ok=True)

    def plot_histogram(data_dict, title, xlabel, filename):
        means = [np.mean(vals) for vals in data_dict.values()]
        mean_of_means = np.mean(means)

        for module_name, vals in sorted(data_dict.items()):
            module_mean = float(np.mean(vals))
            print(f"{title} | Module: {module_name} → Mean = {module_mean:.4f}")

        plt.figure(figsize=(8,6), dpi=300)
        plt.hist(means, bins=10, color="#1F77B4", edgecolor='black', alpha=0.7)
        plt.axvline(mean_of_means, color='red', linestyle='--', label=f"Mean of means = {mean_of_means:.4f}")
        plt.xlabel(xlabel)
        plt.ylabel("Number of Modules")
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"plots/performance/{filename}")
        plt.close()

    plot_histogram(
        per_module_fractional_means,
        "Distribution of per-module fractional RMS improvements",
        "Per-module fractional RMS improvement",
        "histogram_fractional_module_means.pdf"
    )

    plot_histogram(
        per_module_coherent_means,
        "Distribution of per-module coherent noise ratios",
        "Per-module coherent noise ratio (corr/uncorr)",
        "histogram_coherent_module_means.pdf"
    )

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


def _plot_self_cross_hist_for_buckets(buckets, title_prefix, x_label, file_prefix):
    os.makedirs("plots/performance", exist_ok=True)
    BLUE = "#1F77B4"
    ORANGE = "#FF7F0E"

    def _plot_one(drop_key, data_self, data_cross, suffix):
        all_vals = np.array(list(data_self) + list(data_cross))
        if len(all_vals) == 0:
            return

        bins = np.histogram_bin_edges(all_vals, bins="auto")

        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        if len(data_self) > 0:
            plt.hist(data_self, bins=bins, alpha=0.55, edgecolor="black",
                     label=f"SELF (n={len(data_self)})", color=BLUE)
            plt.axvline(np.mean(data_self), linestyle="--", linewidth=1.8, color=BLUE,
                        label=f"SELF mean = {np.mean(data_self):.4f}")
        if len(data_cross) > 0:
            plt.hist(data_cross, bins=bins, alpha=0.55, edgecolor="black",
                     label=f"CROSS (n={len(data_cross)})", color=ORANGE)
            plt.axvline(np.mean(data_cross), linestyle="--", linewidth=1.8, color=ORANGE,
                        label=f"CROSS mean = {np.mean(data_cross):.4f}")

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
        data_self = buckets[drop_key].get("SELF", [])
        data_cross = buckets[drop_key].get("CROSS", [])
        _plot_one(drop_key, data_self, data_cross, f"dropout_{drop_key}")

    all_self = []
    all_cross = []
    for d in buckets.values():
        all_self.extend(d.get("SELF", []))
        all_cross.extend(d.get("CROSS", []))
    _plot_one(None, all_self, all_cross, suffix=None)


def plot_self_cross_validation_histograms():

    # --- Create buckets for Fractional RMS Improvement
    frac_buckets = defaultdict(lambda: {"SELF": [], "CROSS": []})
    for (group_type, dropout_val), vals in dropout_group_results.items():
        frac_buckets[dropout_val][group_type].extend(vals)

    _plot_self_cross_hist_for_buckets(
        buckets=frac_buckets,
        title_prefix="Fractional RMS Improvement (SELF vs CROSS)",
        x_label="Fractional RMS Improvement",
        file_prefix="hist_frac"
    )

    # --- Create buckets for Coherent Noise Ratio
    coh_buckets = defaultdict(lambda: {"SELF": [], "CROSS": []})
    for (group_type, dropout_val), vals in dropout_coherent_group_results.items():
        coh_buckets[dropout_val][group_type].extend(vals)

    _plot_self_cross_hist_for_buckets(
        buckets=coh_buckets,
        title_prefix="Coherent Noise Ratio (corr/uncorr)",
        x_label="Coherent Noise Ratio (corr/uncorr)",
        file_prefix="hist_coh"
    )



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
            print(f"Running evaluation | Evaluate Module: {test_module} | Train Module: {train_module} | Model: {model_name}")

            plot_performance.modulename_for_evaluation = test_module
            plot_performance.train_module = train_module
            plot_performance.new_model_name = model_name

            nodes_per_layer, dropout_rate = parse_model_config(model_name)
            if nodes_per_layer is None or dropout_rate is None:
                print(f"Skipping model {model_name} (invalid format)")
                continue

            plot_performance.nodes_per_layer = nodes_per_layer
            plot_performance.dropout_rate = dropout_rate

            frac_impr_mean, coh_ratio_mean = plot_performance.main()

            register_result(test_module, train_module, nodes_per_layer, dropout_rate,
                            frac_impr_mean, coh_ratio_mean)

        if test_module in per_module_fractional_means:
            mean_frac = np.mean(per_module_fractional_means[test_module])
            print(f"\n Finished {test_module} → Mean Fractional RMS Improvement: {mean_frac:.4f}")
        if test_module in per_module_coherent_means:
            mean_coh = np.mean(per_module_coherent_means[test_module])
            print(f" Finished {test_module} → Mean Coherent Noise Ratio: {mean_coh:.4f}")

    print("\n=====  Fractional RMS Improvement Summary =====")
    for (evaluate_module, train_module, layer_key), values in layer_results.items():
        mean_val = np.mean(values)
        print(
           f"🔎 Summary for Evaluate Module: {evaluate_module}, "
           f"Train Module: {train_module}\n"
           f"  • Layer {layer_key}: Mean Fractional RMS Improvement = {mean_val:.4f}"
        )

    print("\n=====  Detailed Fractional RMS Improvements (Dropout-wise) =====")
    for (evaluate_module, train_module, layer_key), values in detailed_results.items():
       print(f"\n Evaluate: {evaluate_module} | Train: {train_module} | Layer: {layer_key}")
       for dr_value, mean_val in values:
          print(f"  • Dropout {dr_value}: Fractional RMS Improvement = {mean_val:.4f}")

    print("\n=====  Dropout-Based Mean Fractional RMS Improvement =====")

    grouped_results = defaultdict(list)

    for (group_type, dropout_val), values in dropout_group_results.items():
        grouped_results[group_type].append((dropout_val, np.mean(values)))

    for group_type, dropout_list in grouped_results.items():
        print(f"\n {group_type} Models:")
        for dropout_val, mean_val in sorted(dropout_list):
            print(f"  • Dropout {dropout_val}: Mean Fractional RMS Improvement = {mean_val:.4f}")


    print("\n=====  Dropout-Based Mean Coherent Noise Ratio (corr/uncorr) =====")
    coh_grouped_results = defaultdict(list)

    for (group_type, dropout_val), values in dropout_coherent_group_results.items():
        coh_grouped_results[group_type].append((dropout_val, np.mean(values)))

    for group_type, dropout_list in coh_grouped_results.items():
        print(f"\n {group_type} Models:")
        for dropout_val, mean_val in sorted(dropout_list):
            print(f"  • Dropout {dropout_val}: Mean Coherent Noise Ratio = {mean_val:.4f}")


    # =====  Train Module Count-Based Mean Calculations =====
    trained_group_results = defaultdict(list)
    trained_coherent_group_results = defaultdict(list)

    def count_trained_modules(train_module_name):
        return len(re.findall(r'ML_[A-Z0-9]+', train_module_name))

    #Debug layer results
    print("Layer results content:")
    for k, v in layer_results.items():
        print(k, v)

    for (evaluate_module, train_module, layer_key), values in layer_results.items():
        num_trained = count_trained_modules(train_module)
        group_type = "SELF" if is_self(evaluate_module, train_module) else "CROSS"
        print(f"Group: {(group_type, num_trained)} | Values to extend: {values}")
        trained_group_results[(group_type, num_trained)].extend(values)
        trained_coherent_group_results[(group_type, num_trained)].extend(
           coherent_results[(evaluate_module, train_module, layer_key)]
        )

    print("\n=====  Train Module Count-Based Mean Fractional RMS Improvement =====")
    for (group_type, num_trained), vals in trained_group_results.items():
        print(f"{group_type} | Train Modules = {num_trained}: Mean Fractional RMS = {np.mean(vals):.4f}")

    print("\n=====  Train Module Count-Based Mean Coherent Noise Ratio =====")
    for (group_type, num_trained), vals in trained_coherent_group_results.items():
        print(f"{group_type} | Train Modules = {num_trained}: Mean Coherent Noise = {np.mean(vals):.4f}")

    plot_and_save_graphs()
    plot_module_mean_histograms()
    plot_self_cross_validation_histograms()
