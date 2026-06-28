# Downloading the dataset (MVTec LOCO AD)

This project is built around the **MVTec LOCO AD** dataset (Logical and
Structural Anomaly Detection). It is free for academic/research use but
**requires you to fill in a short form** on MVTec's site — there is no direct
`wget` link, so this step is manual.

## Steps

1. Open the dataset page:
   https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad/downloads

2. Accept the licence and download the archive (`mvtec_loco_anomaly_detection.tar`,
   ~ a few GB).

3. **Extract it properly — do not just browse inside it.** On Windows 11 you can
   open a `.tar` in File Explorer and it *looks* like a folder, but the files are
   still inside the archive and the project cannot read them. Right-click the
   `.tar` → **Extract All** (or use 7-Zip), and wait for it to finish. You should
   end up with a real folder containing one folder per product category:

   ```
   breakfast_box/
   juice_bottle/
   pushpins/
   screw_bag/
   splicing_connectors/
   ```

4. Place (or move) those category folders under this repo's `data/mvtec_loco/`
   directory (this is the default `IVP_DATA_DIR`) so the layout looks like:

   ```
   data/mvtec_loco/
     screw_bag/
       train/good/                 <- only "good" images (for training)
       validation/good/            <- good images (for threshold calibration)
       test/
         good/                     <- good test images
         logical_anomalies/        <- e.g. wrong count / wrong part present
         structural_anomalies/     <- e.g. damaged / misplaced part
       ground_truth/               <- pixel masks (not required by this demo)
   ```

   Prefer a different location? Point the project at it instead of moving files:
   ```bash
   export IVP_DATA_DIR=/full/path/to/your/extracted/mvtec_loco
   ```

   > The code also accepts `val/` as an alias for `validation/`.

5. Pick the category you want to work on (default is `screw_bag`):

   ```bash
   export IVP_CATEGORY=screw_bag
   ```

## Why this dataset?

MVTec LOCO is an **anomaly-detection** dataset: you train on *good* parts only,
then flag anything that deviates. That mirrors real factories, where you have
thousands of good units but very few (and unpredictable) defects. It also
distinguishes two defect families that map cleanly onto our `DecisionAgent`:

- **structural** anomalies — a localized physical fault (damage, misplacement);
  these light up a small hot region in the heatmap.
- **logical** anomalies — the part is fine locally but wrong globally (missing
  item, wrong quantity, wrong combination); these show diffuse or no single hot
  spot.

## No dataset yet? You can still run everything

The whole pipeline, database, API and dashboard run **without** the dataset
using the built-in `dummy` backend:

```bash
python scripts/run_demo.py
```

Download the real data only when you are ready to train a real model.
