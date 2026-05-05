"""
DSA 210 - Data Enrichment Script
Feature Engineering + Gaussian Noise Augmentation
"""

import pandas as pd
import numpy as np

# ── 1. Load original data ──────────────────────────────────────────────────
df = pd.read_csv('cleaned_data.csv')
print(f"Original data shape: {df.shape}")

# ── 2. Feature Engineering ────────────────────────────────────────────────
def add_features(data):
    """Derive physically meaningful features from raw print parameters."""
    d = data.copy()

    # Temperature features
    d['heat_ratio']     = d['nozzle_temperature'] / d['bed_temperature']   # nozzle/bed ısı oranı
    d['thermal_delta']  = d['nozzle_temperature'] - d['bed_temperature']   # mutlak ısı farkı

    # Flow/geometry features
    d['volumetric_flow']    = d['layer_height'] * d['print_speed']         # hacimsel akış (mm²/s)
    d['infill_wall_ratio']  = d['infill_density'] / d['wall_thickness']    # doluluk-duvar dengesi

    # Categorical encoding
    d['material_pla']       = (d['material'] == 'pla').astype(int)
    d['pattern_honeycomb']  = (d['infill_pattern'] == 'honeycomb').astype(int)

    return d

df = add_features(df)
print(f"After feature engineering: {df.shape}")

# ── 3. Gaussian Noise Augmentation ───────────────────────────────────────
np.random.seed(42)

# Sütunlar
numerical_input_cols = [
    'layer_height', 'wall_thickness', 'infill_density',
    'nozzle_temperature', 'bed_temperature', 'print_speed', 'fan_speed'
]
output_cols = ['roughness', 'tensile_strength', 'elongation']

NOISE_RATIO = 0.03   # %3 standart sapma → gerçekçi küçük varyasyon
N_COPIES    = 4      # 50 × 4 = 200 yeni satır → toplam 250

augmented_copies = []

for i in range(N_COPIES):
    noisy = df.copy()

    # Input sütunlarına gürültü ekle
    for col in numerical_input_cols:
        std = df[col].std() * NOISE_RATIO
        noise = np.random.normal(0, std, size=len(df))
        noisy[col] = (df[col] + noise).clip(df[col].min(), df[col].max())

    # Output sütunlarına gürültü ekle
    for col in output_cols:
        std = df[col].std() * NOISE_RATIO
        noise = np.random.normal(0, std, size=len(df))
        noisy[col] = (df[col] + noise).clip(df[col].min(), df[col].max())

    # Türetilmiş feature'ları yeniden hesapla (gürültülü inputlara göre)
    noisy['heat_ratio']        = noisy['nozzle_temperature'] / noisy['bed_temperature']
    noisy['thermal_delta']     = noisy['nozzle_temperature'] - noisy['bed_temperature']
    noisy['volumetric_flow']   = noisy['layer_height'] * noisy['print_speed']
    noisy['infill_wall_ratio'] = noisy['infill_density'] / noisy['wall_thickness']

    augmented_copies.append(noisy)

# Orijinal + augmented birleştir
df_final = pd.concat([df] + augmented_copies, ignore_index=True)
df_final = df_final.reset_index(drop=True)

print(f"After augmentation:     {df_final.shape}")

# ── 4. Validate ──────────────────────────────────────────────────────────
print("\n── Distribution check (mean should be similar) ──")
for col in output_cols:
    orig_mean = df[col].mean()
    aug_mean  = df_final[col].mean()
    print(f"  {col:20s}  original mean: {orig_mean:.2f}  |  enriched mean: {aug_mean:.2f}")

print(f"\nNew features added: heat_ratio, thermal_delta, volumetric_flow, infill_wall_ratio, material_pla, pattern_honeycomb")
print(f"Total columns: {df_final.shape[1]}")

# ── 5. Save ───────────────────────────────────────────────────────────────
df_final.to_csv('enriched_data.csv', index=False)
print(f"\n✅ Saved: enriched_data.csv  ({df_final.shape[0]} rows × {df_final.shape[1]} cols)")
