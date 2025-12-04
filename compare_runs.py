import os
import json
import pandas as pd
import numpy as np

ABLATION_CSV = "ablation_runs.csv"
SUMMARY_CSV = "ablation_summary.csv"


def safe_last_row(df, split_value=None, subset_cols=None):
    """
    Return the last row (optionally for a given split) that has
    non-null values in `subset_cols`. If none, return None.
    """
    if split_value is not None and "split" in df.columns:
        df = df[df["split"] == split_value]

    if df.empty:
        return None

    if subset_cols is not None:
        df = df.dropna(subset=subset_cols, how="any")
        if df.empty:
            return None

    return df.iloc[-1]


def safe_float(x):
    if x is None:
        return np.nan
    try:
        if pd.isna(x):
            return np.nan
    except TypeError:
        pass
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def read_last_train_metrics(csv_path):
    if not os.path.exists(csv_path):
        print(f"  train_log missing at {csv_path}")
        return {"train_mse": np.nan, "train_policy_loss": np.nan}

    df = pd.read_csv(csv_path)
    # last train row with non-NaN mse & policy_loss
    last = safe_last_row(df, split_value="train", subset_cols=["mse_loss", "policy_loss"])
    if last is None:
        return {"train_mse": np.nan, "train_policy_loss": np.nan}

    return {
        "train_mse": safe_float(last.get("mse_loss")),
        "train_policy_loss": safe_float(last.get("policy_loss")),
    }


def read_last_val_metrics(csv_path):
    if not os.path.exists(csv_path):
        print(f"  val_log missing at {csv_path}")
        return {"val_crps": np.nan, "val_mse": np.nan, "val_policy_loss": np.nan}

    df = pd.read_csv(csv_path)
    # val_log already only has val rows, but we filter by split for safety
    last = safe_last_row(df, split_value="val_epoch", subset_cols=["crps", "mse_loss", "policy_loss"])
    if last is None:
        return {"val_crps": np.nan, "val_mse": np.nan, "val_policy_loss": np.nan}

    return {
        "val_crps": safe_float(last.get("crps")),
        "val_mse": safe_float(last.get("mse_loss")),
        "val_policy_loss": safe_float(last.get("policy_loss")),
    }


def read_test_metrics(csv_path):
    if not os.path.exists(csv_path):
        print(f"  test_log missing at {csv_path}")
        return {"test_crps": np.nan, "test_mse": np.nan, "test_policy_loss": np.nan}

    df = pd.read_csv(csv_path)
    # test_log has a single "final" row; take last as a safeguard
    last = df.iloc[-1]
    return {
        "test_crps": safe_float(last.get("crps")),
        "test_mse": safe_float(last.get("mse_loss")),
        "test_policy_loss": safe_float(last.get("policy_loss")),
    }


def parse_hparams_from_name_and_overrides(name, overrides_json_str):
    lam = None
    value_coef = None
    entropy_beta = None

    # 1) try from run "name", e.g. "lam0.80_v0.5_beta0.05"
    tokens = name.split("_")
    for tok in tokens:
        if tok.startswith("lam"):
            try:
                lam = float(tok[3:])
            except ValueError:
                pass
        elif tok.startswith("v"):
            # v0.5
            try:
                value_coef = float(tok[1:])
            except ValueError:
                pass
        elif tok.startswith("beta"):
            try:
                entropy_beta = float(tok[4:])
            except ValueError:
                pass

    # 2) fallback to overrides_json if needed
    if overrides_json_str and (lam is None or value_coef is None or entropy_beta is None):
        try:
            overrides = json.loads(overrides_json_str)
            # overrides like "algorithm.schedule_matrix.lam=0.80"
            for ov in overrides:
                if "algorithm.schedule_matrix.lam=" in ov and lam is None:
                    lam = float(ov.split("=")[-1])
                elif "algorithm.schedule_matrix.value_coef=" in ov and value_coef is None:
                    value_coef = float(ov.split("=")[-1])
                elif "algorithm.schedule_matrix.entropy_beta=" in ov and entropy_beta is None:
                    entropy_beta = float(ov.split("=")[-1])
        except Exception:
            pass

    return lam, value_coef, entropy_beta


def main():
    if not os.path.exists(ABLATION_CSV):
        print(f"{ABLATION_CSV} not found in current directory.")
        return

    ablation_df = pd.read_csv(ABLATION_CSV)

    results = []

    for _, row in ablation_df.iterrows():
        run_dir = row["run_dir"]         # e.g. "outputs/2025-12-04/01-09-28"
        name = row.get("name", os.path.basename(run_dir))
        overrides_json_str = row.get("overrides_json", "")

        print(f"Processing run: {os.path.basename(run_dir)}")

        if not os.path.isdir(run_dir):
            print(f"  Run directory does not exist: {run_dir}")
            continue

        train_csv = os.path.join(run_dir, "training_log", "train_log.csv")
        val_csv = os.path.join(run_dir, "training_log", "val_log.csv")
        test_csv = os.path.join(run_dir, "training_log", "test_log.csv")

        train_metrics = read_last_train_metrics(train_csv)
        val_metrics = read_last_val_metrics(val_csv)
        test_metrics = read_test_metrics(test_csv)

        lam, value_coef, entropy_beta = parse_hparams_from_name_and_overrides(
            name, overrides_json_str
        )

        results.append(
            {
                "run_dir": run_dir,
                "name": name,
                "lam": lam,
                "value_coef": value_coef,
                "entropy_beta": entropy_beta,
                **train_metrics,
                **val_metrics,
                **test_metrics,
            }
        )

    if not results:
        print("No runs processed.")
        return

    summary_df = pd.DataFrame(results)

    # sort by validation CRPS (lower is better)
    summary_df = summary_df.sort_values(by="val_crps", ascending=True)

    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved summary to {SUMMARY_CSV}\n")

    # print a quick view
    cols_to_show = [
        "name",
        "lam",
        "value_coef",
        "entropy_beta",
        "val_crps",
        "val_mse",
        "val_policy_loss",
        "test_crps",
        "test_mse",
        "test_policy_loss",
    ]
    print(summary_df[cols_to_show].to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
