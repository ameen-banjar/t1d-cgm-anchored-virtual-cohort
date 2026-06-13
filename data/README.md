# Data policy

This repository does not redistribute participant-level CGM records.

`data/derived/virtual_profile_summary.csv` contains 180 simulator-derived
profile summaries. These are virtual cohort members, not real participants.
To reproduce the analysis:

1. Obtain T1DiabetesGranada under the provider's terms.
2. Create `data/raw/real_summary.csv` using the documented column schema.
3. Place participant traces in `data/raw/real_cgm/`.
4. Place Simglucose traces in `data/raw/virtual_traces/`.
5. Run `bash scripts/run_full_analysis.sh`.

The committed outputs are aggregate statistics only.
