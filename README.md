# Machine Learning Algorithm for Optical Filters

## What Does This Do?

This project combines **physics-based simulation** with **machine learning** to analyze and predict the optical properties of multi-layer thin film filters. It implements the **Transfer Matrix Method (TMM)** to calculate how light interacts with layered optical structures, and uses neural networks to learn patterns in optical filter behavior.

## Overview

Optical filters are essential components in many technologies including displays, sensors, solar cells, and optical devices. This project provides tools to:

1. **Simulate optical properties** of multi-layer thin film stacks using physics-based calculations
2. **Predict filter behavior** using machine learning models trained on simulation data
3. **Design custom filter stacks** through an interactive interface

## Key Components

### 1. Transfer Matrix Method Simulation (`main.py`)

The core physics simulation that calculates optical properties of thin film stacks.

**What it does:**
- Uses the Transfer Matrix Method (TMM) to model light propagation through multiple layers
- Calculates **reflectance** (how much light bounces back) and **transmittance** (how much light passes through)
- Works across the visible spectrum (wavelengths from ~400-800 nm)
- Visualizes results with dual-axis plots showing both reflectance and transmittance

**How it works:**
1. Takes a stack of materials with specified thicknesses
2. For each wavelength, calculates:
   - Fresnel reflection and transmission coefficients at each interface
   - Phase changes as light propagates through each layer
   - Overall reflectance and transmittance of the entire stack
3. Plots the spectral response

**Example stack in the code:**
```python
# Variable name in main.py is 'layers'
layers = [
    ['air', 1000],           # Incident medium
    ['quartz', 5],           # 5 nm quartz layer
    ['glass', 25],           # 25 nm glass layer
    ['indium tin oxide', 15],# 15 nm ITO layer
    ['copper', 3],           # 3 nm copper layer
    ['air', 100]             # Transmission medium
]
```

### 2. Machine Learning Model (`ai (1).py`)

An incomplete neural network implementation intended to predict optical filter properties.

**What it's designed to do:**
- Train a deep learning model to predict reflectance and transmittance
- Input features: wavelength, material properties, layer thicknesses
- Output: optical properties (reflectance and transmittance)
- Uses Keras/TensorFlow for the neural network

**Current status:** 
- The code is a template/work-in-progress
- Contains placeholder references to housing data that need to be replaced with optical filter data
- Requires dataset generation from TMM simulations to train the model

**Intended workflow:**
1. Generate training data using TMM simulations with various material combinations
2. Train neural network on this data
3. Use trained model to quickly predict properties of new filter designs without running full simulations

### 3. Interactive Layer Builder (`inputs.py`)

A simple command-line interface for building filter stacks.

**What it does:**
- Prompts user to select materials from available options
- Allows input of layer thickness for each material
- Builds up a layer stack interactively
- Validates inputs and confirms selections

**Materials available:** low, high, mid (referring to refractive index categories)

**Note:** The materials in `inputs.py` are simplified placeholders. The actual material database (`refractive-indexB.csv`) contains comprehensive optical data for many materials including: silver, aluminium, aluminium oxide, gold, glass, cobalt, chromium, copper, gallium nitride, indium tin oxide, magnesium oxide, nickel, polyethylene, platinum, quartz, silicon, silicon nitride, silicon oxide, titanium, tin oxide, tungsten, air, and various polymers (PVA, PVC, PVP, PMMA).

### 4. Data Files

- **`refractive-indexB.csv`**: Database of refractive indices for various materials across different wavelengths
- **`UV spectrometry.xlsx`** and **`UV spectrometry - Copy.xlsx`**: Experimental UV-Vis spectroscopy data
- **`Spin Coating Tables.xlsx`**: Reference data for thin film deposition parameters
- **`material database.zip`**: Additional material property data

## How to Use

### Running the TMM Simulation

```bash
python main.py
```

This will:
1. Load the refractive index database
2. Simulate the predefined layer stack
3. Calculate reflectance and transmittance across wavelengths
4. Display plots showing the optical response

### Using the Layer Builder

```bash
python inputs.py
```

Follow the prompts to:
1. Select materials
2. Enter thicknesses
3. Build your custom filter stack

## Physics Background

### Transfer Matrix Method (TMM)

TMM is a powerful technique for analyzing light propagation in stratified media:

1. **Boundary Matrices (D)**: Describe reflection and transmission at interfaces using Fresnel equations
2. **Propagation Matrices (P)**: Account for phase changes as light travels through a layer
3. **System Matrix (M)**: Product of all boundary and propagation matrices
4. **Optical Properties**: Calculated from the system matrix elements

The method accounts for:
- Multiple reflections between layers (interference effects)
- Phase relationships (constructive/destructive interference)
- Material dispersion (wavelength-dependent refractive indices)

## Applications

This tool can be used for:
- **Optical filter design**: Anti-reflection coatings, dichroic filters, beam splitters
- **Display technology**: Color filters, brightness enhancement films
- **Solar cells**: Anti-reflection and light-trapping structures
- **Optical sensors**: Wavelength-selective detectors
- **Research**: Understanding thin-film interference effects

## Requirements

```
numpy
pandas
matplotlib
seaborn
keras/tensorflow (for machine learning component)
scikit-learn (for machine learning component)
```

## Future Development

The project aims to integrate TMM simulation with machine learning:
1. Generate large training datasets using TMM
2. Train neural networks to learn material-property-performance relationships
3. Use ML models for rapid design space exploration
4. Optimize filter designs for specific target spectra

## Author Attribution

The machine learning template (`ai (1).py`) is based on work by Sreenivas Bhattiprolu (Python for Microscopists).

## License

Feel free to copy and use. Acknowledgment of sources is appreciated.
