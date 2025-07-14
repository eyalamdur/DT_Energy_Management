# Results Directory

This folder contains large data outputs that have been compressed to save space. Before using them, you need to extract the contents of the ZIP archives.

## trajectories

- **Location:** `results/trajectories`  
- Contains individual `.zip` files for each trajectory (originally `.pkl`).

**Extract all trajectory archives in-place:**

```bash
find results/trajectories -type f -name "*.zip" -execdir unzip {} \;
```

## evaluate

- **Location:** `results/evaluate`  
- Contains date-named subfolders. Each subfolder contains one or more `.zip` archives of evaluation outputs (the folders themselves are not zipped).

**Extract all evaluation archives in-place (within each date folder):**

```bash
find results/evaluate -type f -name "*.zip" -execdir unzip {} \;
```

Once extracted, the raw `.pkl` or other output files will appear alongside their corresponding ZIP archives in each folder. Ensure you have sufficient disk space before unzipping large archives.