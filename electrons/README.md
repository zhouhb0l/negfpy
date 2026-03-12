# Electron Utilities

This folder is reserved for electron-only work.

It is intentionally separate from the phonon-focused `src/negfpy` package.

Code stays here. Analysis outputs go to `14. TaN/electronic properties`.

## Scope

- Read electron transmission data from Nanodcal `CalculatedResults.json`
- Work only with electron transmission channels
- Do not mix phonon workflows into this folder

## Scripts

- `inspect_electron_transmission.py`
- `plot_electron_transmission.py`
- `plot_electron_conductance.py`
- `plot_electron_analysis.py`

Example:

```powershell
python electrons/inspect_electron_transmission.py "C:/Users/zhbho/OneDrive/Desktop/workspace/NSCCscratch/DeviceStudioProject/TaN/TaN/Nanodcal-Crystal/ElectronTransChannel_direction_1/CalculatedResults.json"
```

Plot transmission and transmission per area:

```powershell
python electrons/plot_electron_transmission.py "C:/Users/zhbho/OneDrive/Desktop/workspace/NSCCscratch/DeviceStudioProject/TaN/4Mg25/Nanodcal-Crystal/ElectronTransChannel_direction_1"
```

Default output location:

```text
14. TaN/electronic properties/electrons/<case>/<ElectronTransChannel_direction_*>
```

Plot electrical and electronic thermal conductance versus temperature:

```powershell
python electrons/plot_electron_conductance.py "C:/Users/zhbho/OneDrive/Desktop/workspace/NSCCscratch/DeviceStudioProject/TaN/4Mg25/Nanodcal-Crystal/ElectronTransChannel_direction_1"
```

Create comparison plots for all materials in x, y, and z:

```powershell
python electrons/plot_electron_analysis.py
```
