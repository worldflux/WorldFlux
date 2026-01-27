# Models

## DreamerV3

RSSM-based world model with categorical latent space.

### Architecture

- **Encoder**: CNN (images) or MLP (states)
- **RSSM**: GRU deterministic + categorical stochastic
- **Decoder**: Reconstructs observations, rewards, continues

### Presets

| Preset | Params | deter_dim | stoch_dim |
|--------|--------|-----------|-----------|
| size12m | ~12M | 2048 | 1024 |
| size25m | ~25M | 2048 | 1024 |
| size50m | ~50M | 4096 | 1024 |
| size100m | ~100M | 4096 | 2048 |
| size200m | ~200M | 8192 | 2048 |

### Usage

```python
model = create_world_model("dreamerv3:size12m", obs_shape=(3, 64, 64), action_dim=4)

state = model.encode(obs)
print(state.deterministic.shape)  # [B, 2048]
print(state.stochastic.shape)     # [B, 1024]

# Can decode imagined states
decoded = model.decode(state)
print(decoded["obs"].shape)  # [B, 3, 64, 64]
```

### Best For

- Image observations
- Atari games
- Tasks requiring observation reconstruction

---

## TD-MPC2

Implicit world model with SimNorm latent space.

### Architecture

- **Encoder**: MLP with SimNorm
- **Dynamics**: MLP (no decoder)
- **Q-Ensemble**: Multiple Q-networks for planning

### Presets

| Preset | Params | latent_dim | num_q |
|--------|--------|------------|-------|
| 5m | ~5M | 512 | 5 |
| 19m | ~19M | 512 | 5 |
| 48m | ~48M | 1024 | 5 |
| 317m | ~317M | 2048 | 10 |

### Usage

```python
model = create_world_model("tdmpc2:5m", obs_shape=(39,), action_dim=6)

state = model.encode(obs)
print(state.deterministic.shape)  # [B, 512]
print(state.stochastic)           # None

# Q-values for planning
q_values = model.q(state, action)
```

### Best For

- State vector observations
- MuJoCo / continuous control
- MPC-style planning

---

## Comparison

| | DreamerV3 | TD-MPC2 |
|---|---|---|
| Input | Images or states | States |
| Latent | Categorical | SimNorm |
| Decoder | Yes | No |
| Best for | Visual tasks | Continuous control |
