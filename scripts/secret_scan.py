#!/usr/bin/env python3
from pathlib import Path
import re


PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(
        r"(?i)(password|passwd|secret|token)\s*=\s*[\"'][^\"']+[\"']"
    ),
    re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9._-]+\\"),
]
IGNORE = {".git", ".venv", "__pycache__", "outputs", "manuscript", "release"}
findings = []
for path in Path(".").rglob("*"):
    if any(part in IGNORE for part in path.parts) or not path.is_file():
        continue
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".zip", ".gz"}:
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    if any(pattern.search(text) for pattern in PATTERNS):
        findings.append(str(path))
if findings:
    raise SystemExit("Potential secrets found:\n" + "\n".join(findings))
print("No obvious secrets found.")
