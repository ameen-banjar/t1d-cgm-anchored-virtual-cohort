# Input schema

## Real summary

Required canonical columns:

```text
subject_id, mean_glucose_mgdl, cv_percent, tir_percent
```

Recommended validation columns:

```text
gmi_percent, tbr_percent, tar_percent, lbgi, hbgi
```

The reader also accepts legacy analysis names such as `Real_Mean`,
`Real_CV`, and `Real_Patient_ID`.

## Virtual summary

Required canonical columns:

```text
virtual_id, scenario, mean_glucose_mgdl, cv_percent, tir_percent
```

Recommended validation columns:

```text
gmi_percent, tbr_percent, tar_percent, lbgi, hbgi, trace_path
```

## Privacy

Subject-level match and diurnal files are written to `outputs/private/`,
which is excluded by `.gitignore`. Only aggregate tables and figures should
be committed.
