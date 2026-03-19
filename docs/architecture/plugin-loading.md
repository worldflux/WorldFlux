# Plugin Loading Sequence

External plugins register model families and components through Python
entry points. This diagram shows the discovery and registration flow.

```mermaid
sequenceDiagram
    participant Caller
    participant Registry as WorldModelRegistry
    participant EP as importlib.metadata
    participant Plugin as Plugin Module
    participant Manifest as PluginManifest

    Caller->>Registry: list_models() / from_pretrained()
    Registry->>Registry: load_entrypoint_plugins()

    alt not _plugins_loaded
        Registry->>EP: entry_points().select(group="worldflux.models")
        EP-->>Registry: [EntryPoint, ...]

        loop for each entry_point
            Registry->>Plugin: entry_point.load()
            Plugin-->>Registry: module / callable

            alt callable returns manifest
                Plugin-->>Registry: PluginManifest / dict
                Registry->>Manifest: _normalize_plugin_manifest()
                Registry->>Manifest: _validate_plugin_manifest()
                Manifest-->>Registry: validated PluginManifest
            end

            alt no manifest provided
                Registry->>Registry: register_plugin_manifest(name, default)
            end

            Registry->>Registry: _validate_plugin_manifest(name, manifest)

            alt version mismatch
                Registry-->>Caller: ConfigurationError
            end
        end

        Registry->>EP: entry_points().select(group="worldflux.components")
        Note over Registry: Same flow for component plugins

        Registry->>Registry: _plugins_loaded = True
    end

    Registry-->>Caller: registered models / components
```

## Entry Point Groups

| Group | Purpose | Example |
|-------|---------|---------|
| `worldflux.models` | Register model families | `my_model = my_pkg:register` |
| `worldflux.components` | Register reusable components | `my_encoder = my_pkg:register` |

## Manifest Validation

Plugins must declare compatibility via `PluginManifest`:

- `plugin_api_version`: Must be non-empty (currently `"0.x-experimental"`)
- `worldflux_version_range`: PEP 440 specifier checked against runtime
- `experimental`: Must be `True` in current API phase
- `capabilities`: Tuple of capability strings

Invalid manifests raise `ConfigurationError` at load time.
