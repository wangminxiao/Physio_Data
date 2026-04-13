# Workzone

Remote server working directory. Code and configs here are meant to be run on the
server where the raw data lives.

## Structure

```
workzone/
├── mimic3/              # MIMIC-III extraction scripts
├── mcmed/               # MC_MED conversion scripts
├── configs/             # Server-specific configs (data paths, worker counts)
├── logs/                # Extraction logs and verification reports
└── outputs/             # Small outputs to bring back (profiles, manifests, plots)
```

## Workflow

1. Claude writes scripts locally -> git push
2. On server: git pull -> run scripts
3. Commit small outputs (logs, manifests) -> git push
4. Locally: git pull -> Claude reviews results

Large outputs (processed .npy files) stay on the server and are .gitignored.
