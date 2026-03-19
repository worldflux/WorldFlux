# API Stability

WorldFlux classifies public Python surfaces into three stability tiers:

- `stable`: safe to build against across minor releases
- `experimental`: public, but subject to faster iteration and shape changes
- `internal`: not part of the supported public contract

## Current Policy

- Top-level APIs such as `create_world_model()` and core payload/config/spec
  types are treated as `stable`.
- Backend-handle and backend-routing surfaces are currently `experimental`.
- Concrete family implementation classes are currently `experimental` even when
  their factory entry points are public.

## Machine-Readable Manifest

Generate the current manifest with:

```bash
python scripts/generate_public_api_manifest.py
```

Verify that the checked-in manifest matches the current source tree with:

```bash
python scripts/generate_public_api_manifest.py --check
```

The manifest schema version is `worldflux.public_api_manifest.v1`.
