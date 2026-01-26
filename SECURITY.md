# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it by emailing the maintainers directly. Do not open a public issue.

We will respond within 48 hours and work with you to understand and address the issue.

## Security Considerations

### Model Checkpoints

**WARNING**: Loading model checkpoints from untrusted sources can execute arbitrary code.

- `torch.load()` uses Python's pickle module internally, which can execute arbitrary code during deserialization
- Only load checkpoints from sources you trust
- The `Trainer.load_checkpoint()` method requires `weights_only=False` to load optimizer states

**Safe practices:**
```python
# Only load from trusted sources
trainer.load_checkpoint("path/to/trusted/checkpoint.pt")

# For model weights only (safer), use the registry
model = AutoWorldModel.from_pretrained("path/to/model")  # uses weights_only=True
```

### Replay Buffer Data

- `ReplayBuffer.load()` uses `np.load()` with `allow_pickle=False` for security
- Only NumPy array data is loaded, not arbitrary Python objects
- This is safe to use with data from any source

### File Path Handling

- All file operations use `pathlib.Path` for proper path handling
- No shell commands are executed with user-provided paths
- Relative paths are resolved relative to the current working directory

## Dependencies

All dependencies are from trusted sources:
- PyTorch (BSD License)
- NumPy (BSD License)
- tqdm (MIT/MPL-2.0 dual license)
- gymnasium (MIT License)
- wandb (MIT License)

## Best Practices

1. **Validate data sources**: Only load checkpoints and data from trusted sources
2. **Use virtual environments**: Isolate your project dependencies
3. **Keep dependencies updated**: Regularly update to get security patches
4. **Review third-party models**: Inspect pretrained models before loading
