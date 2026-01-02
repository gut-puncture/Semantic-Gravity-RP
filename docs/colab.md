# Running the Experiment in Colab

This guide covers running the full Stage 2 GPU pipeline from `notebooks/semantic_gravity.ipynb`.

## Prerequisites

- Google Colab with an A100 (or similar) GPU runtime
- Google Drive space for model files and outputs
- `data/validated/*.jsonl` available in your Drive copy of the repo

## Recommended Drive Layout

Default paths used by the notebook (you can change them):

```
/content/drive/MyDrive/Semantic-Gravity-RP/
  notebooks/semantic_gravity.ipynb
  src/
  data/
    validated/
    validated_single_token/   (optional; notebook can create this)
    prompts.csv               (optional; notebook can create this)
  outputs/                    (created by the notebook)

/content/drive/MyDrive/models/Qwen2.5-7B-Instruct/
```

## Steps

1. Open Colab and enable a GPU runtime.
2. Copy this repo into Drive (upload the folder or zip it and unzip in Drive).
3. Ensure `data/validated/*.jsonl` is present in the repo folder on Drive.
4. Place the model in Drive or point to a local HF cache path.
5. Open `notebooks/semantic_gravity.ipynb` from Drive and run cells top to bottom.

## Updating Drive Locations

Edit the configuration cell near the top of the notebook to match your Drive paths:

- `SEMANTIC_GRAVITY_ROOT`: root folder containing the repo
- `SEMANTIC_GRAVITY_MODEL_PATH`: model folder or HF model id
- `SEMANTIC_GRAVITY_DATA_ROOT` (optional): defaults to `<root>/data`
- `SEMANTIC_GRAVITY_OUTPUT_ROOT` (optional): defaults to `<root>/outputs`

Example overrides:

```
os.environ["SEMANTIC_GRAVITY_ROOT"] = "/content/drive/MyDrive/Semantic-Gravity-RP"
os.environ["SEMANTIC_GRAVITY_MODEL_PATH"] = "/content/drive/MyDrive/models/Qwen2.5-7B-Instruct"
```

The notebook prints the resolved paths before running the pipeline.

## Outputs

Results are written under:

```
<root>/outputs/experiment_run_YYYYMMDD_HHMM/
```

Key artifacts include:
- `runs/completions_*.jsonl`
- `runs/detection_mapping*.jsonl`
- `runs/attention_metrics.csv`
- `runs/logit_lens.csv`
- `runs/ffn_attn_decomp.csv`
- `runs/patching_results.csv`
- `runs/bootstrap_results.csv`
- `figures/*.png`

## Troubleshooting

- If outputs are in the wrong place, check the printed `Project root` and `Output root`.
- If you see NumPy or dependency errors, restart the runtime and rerun the install cell first.
