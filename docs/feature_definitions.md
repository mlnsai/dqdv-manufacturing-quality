# dQ/dV Feature Definitions

This document describes the ten features extracted from differential voltage (dQ/dV) analysis of lithium-ion cells with **graphite || NMC/NCA** chemistry.

## Overview

Each cell yields **five peaks** from formation-cycle dQ/dV curves — three during charge (CP1–CP3) and two during discharge (DP1–DP2). For each peak, two parameters are extracted:

- **Voltage position** (V): the voltage at which the peak maximum occurs
- **Peak intensity** (Ah/V): the magnitude of dQ/dV at the peak maximum

This gives **10 features per cell**.

---

## Charge Peaks

### CP1 — Low-Voltage Charge Peak
- **Typical voltage**: ~3.45–3.50 V (full cell)
- **Electrochemical origin**: Onset of lithium intercalation into graphite; transition from dilute stage (Stage 1L → Stage 4)
- **Sensitivity**: Moderate manufacturing variability; affected by electrolyte wetting quality and formation protocol

### CP2 — Mid-Voltage Charge Peak
- **Typical voltage**: ~3.54–3.58 V (full cell)
- **Electrochemical origin**: Intermediate graphite staging transitions (Stage 3 → Stage 2)
- **Sensitivity**: Typically the smallest peak; high relative variability (CV%) makes it a sensitive manufacturing indicator

### CP3 — Main Charge Peak
- **Typical voltage**: ~3.63–3.67 V (full cell)
- **Electrochemical origin**: Dominant capacity-delivering process — graphite Stage 2 → Stage 1 transition, coinciding with NMC cathode phase transition (H2 → H3 in layered oxides)
- **Sensitivity**: Largest peak; its intensity directly correlates with usable cell capacity

---

## Discharge Peaks

### DP1 — Main Discharge Peak (Low-Voltage)
- **Typical voltage**: ~3.40–3.44 V (full cell)
- **Electrochemical origin**: Reverse of CP1 — lithium de-intercalation from graphite, early staging reversal
- **Sensitivity**: Forms a charge–discharge pair with CP1; voltage gap (hysteresis) indicates kinetic losses

### DP2 — High-Voltage Discharge Peak
- **Typical voltage**: ~3.56–3.60 V (full cell)
- **Electrochemical origin**: Reverse of CP3 — dominant discharge process corresponding to Stage 1 → Stage 2 graphite de-staging and cathode reduction
- **Sensitivity**: Paired with CP3; their intensity ratio is a proxy for coulombic efficiency

---

## Derived Metrics

These are computed from the 10 primary features during analysis:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Voltage hysteresis | ΔV = V_charge − V_discharge | Kinetic polarization; lower is better |
| Intensity ratio | R = I_charge / I_discharge | Reversibility; ≈1.0 is ideal |
| Health score | (CP3_intensity + DP2_intensity) / 2 | Composite capacity proxy |
| CV% | σ/μ × 100% | Manufacturing variability per feature |

---

## Notes on Chemistry Adaptation

The peak assignments above assume **graphite anode** and **NMC or NCA cathode** chemistry. For other systems:

- **LFP cathode**: Expect a single dominant flat peak pair (due to two-phase olivine reaction); fewer peaks overall
- **Silicon-blend anode**: Additional peaks from Si lithiation (~0.1–0.4 V vs Li/Li⁺) may appear
- **LTO anode**: Very different voltage window (~2.4 V vs Li/Li⁺); peak assignments must be rederived

The statistical framework (Phases 1–3) is chemistry-agnostic; only the physical interpretation (Phase 4) requires adaptation.
