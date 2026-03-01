# Shared Assets

This directory stores reusable project assets that may be mounted or copied into deployments.

## Contents

- `shared/models/models/`: model artifacts (GGUF, embedding model, adapters)
- `shared/data/data/`: source documents and prebuilt retrieval indexes
- `shared/config/config/`: example configuration module
- `shared/docs/`: deployment notes

Runtime code in `src/backend` automatically falls back to these paths when root-level
`models/` or `data/` directories are not present.
