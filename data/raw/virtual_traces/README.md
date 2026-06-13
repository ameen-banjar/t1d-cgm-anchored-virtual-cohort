# Virtual traces

Place the six scenario folders containing the 180 Simglucose CSV traces
under this directory. The pipeline recursively indexes the files by virtual
subject identifier and scenario folder.

The committed `data/derived/virtual_profile_summary.csv` is sufficient for
whole-day matching. The time-resolved analysis requires the full traces.
They contain simulator output only and may be distributed through Zenodo;
they are excluded from Git because the collection is approximately 719 MB.
