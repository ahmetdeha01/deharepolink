# DSA210 Project - 3D Printing Parameter Analysis

# DSA 210: Introduction to Data Science - Term Project

**Term:** 2025-2026 Spring  
**Student:** Ahmet Deha Yıldırım
**Student ID:** 00032656  

**Project Title:** 3D Printing Parameter Analysis

## Project Overview
This project investigates how 3D printing parameters affect print quality and mechanical performance. The analysis focuses on how controllable printing settings affect three output variables: surface roughness, tensile strength, and elongation.

## Dataset
The main dataset contains experimental observations from 3D printing processes. The input variables include:
- layer_height
- wall_thickness
- infill_density
- infill_pattern
- nozzle_temperature
- bed_temperature
- print_speed
- material
- fan_speed

The output variables are:
- roughness
- tensile_strength
- elongation

## Project Structure
- `data/raw/` : original dataset files
- `data/processed/` : cleaned dataset files
- `notebooks/` : Colab notebooks

## Current Progress
For the 14 April milestone, the following steps have been completed:
- Dataset collected
- Data cleaning performed
- Exploratory data analysis completed
- Initial hypothesis tests conducted

## Initial Hypotheses
1. Higher layer height is associated with increased surface roughness.
2. Higher infill density is associated with higher tensile strength.
3. Material type has a significant effect on elongation.

## Files
- `data/raw/data.csv`
- `data/processed/cleaned_data.csv`
- `notebooks/01_data_cleaning_eda.ipynb`
- `dsa210_ahmetdeha_3dprint_proposal.pdf`

## Next Step
The next stage of the project will focus on applying machine learning methods to predict roughness, tensile strength, and elongation from printing parameters.
