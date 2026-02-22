# WorldFlux Demo Recording Guide

## Required Tools

| Tool | Version | Install |
|------|---------|---------|
| asciinema | >= 3.0 | `brew install asciinema` or `pip install asciinema` |
| agg | latest | `brew install asciinema/tap/agg` or [GitHub releases](https://github.com/asciinema/agg/releases) |
| ffmpeg | >= 5.0 | `brew install ffmpeg` |
| Python | >= 3.10 | Required for worldflux |

## Terminal Settings

Configure your terminal before recording:

| Setting | Value |
|---------|-------|
| Font | JetBrains Mono, 18px |
| Background | `#1a1a2e` (dark navy) |
| Foreground | `#e0e0e0` |
| Resolution | 1920x1080 (16:9) |
| Columns | 100 (90s) / 80 (30s) |
| Rows | 30 (90s) / 24 (30s) |
| Cursor | Block, blinking |
| Scrollback | Disabled during recording |

### iTerm2 Setup

1. Preferences > Profiles > Text > Font: JetBrains Mono 18
2. Preferences > Profiles > Colors > Background: `#1a1a2e`
3. Preferences > Profiles > Window > Columns: 100, Rows: 30

### Alacritty Setup

```toml
[font]
normal = { family = "JetBrains Mono", style = "Regular" }
size = 18.0

[colors.primary]
background = "#1a1a2e"
foreground = "#e0e0e0"

[window]
dimensions = { columns = 100, lines = 30 }
```

## Recording Specifications

| Spec | 90-Second Demo | 30-Second Short |
|------|---------------|-----------------|
| Script | `record_demo.sh` | `record_short.sh` |
| Cast file | `worldflux-demo-90s.cast` | `worldflux-demo-30s.cast` |
| Duration | ~90 seconds | ~30 seconds |
| Columns x Rows | 100x30 | 80x24 |
| Target format | MP4 (YouTube/Twitter) | GIF (GitHub README) |
| File size target | < 20 MB (MP4) | < 5 MB (GIF) |

## 90-Second Demo Timeline

| Timestamp | Section | Content |
|-----------|---------|---------|
| 0:00-0:05 | Title | WorldFlux branding, tagline |
| 0:05-0:15 | Install | `pip install worldflux` |
| 0:15-0:30 | Scaffold | `worldflux init demo-project` with MuJoCo defaults |
| 0:30-0:45 | Explore | `ls` generated files, `head worldflux.toml` |
| 0:45-1:00 | Verify | `worldflux verify` with proof-grade parity output |
| 1:00-1:15 | Badge | `worldflux parity badge` SVG generation |
| 1:15-1:30 | CTA | GitHub link, docs link, star call-to-action |

## 30-Second Short Timeline

| Timestamp | Section | Content |
|-----------|---------|---------|
| 0:00-0:03 | Title | WorldFlux branding |
| 0:03-0:08 | Install | `pip install worldflux` |
| 0:08-0:18 | Verify | Parity verification -> PASS |
| 0:18-0:23 | Badge | Badge generation |
| 0:23-0:30 | CTA | GitHub + install links |

## Handling `worldflux init` Interactive Prompts

The `worldflux init` command uses InquirerPy (arrow-key selection menus) or
falls back to Rich prompts (line-by-line stdin).  There is **no**
`--non-interactive` flag.

### Approach 1: Pipe stdin (Rich fallback)

When InquirerPy is not importable, the CLI falls back to `rich.prompt.Prompt`
which reads from stdin line-by-line.  The demo script uses this approach:

```bash
# Temporarily make InquirerPy un-importable, or use an env without it
printf '%s\n' \
    "demo-project" \
    "mujoco" \
    "39" \
    "6" \
    "tdmpc2:ci" \
    "100000" \
    "16" \
    "n" \
    "y" \
| worldflux init ./demo-project --force
```

The prompts in order:
1. Project name (default: `my-world-model`)
2. Environment type: `atari` | `mujoco` | `custom`
3. Observation shape (comma-separated, e.g. `39`)
4. Action dimension (positive int, e.g. `6`)
5. Model choice: `dreamer:ci` | `tdmpc2:ci`
6. Total training steps (default: `100000`)
7. Batch size (default: `16`)
8. Prefer GPU? `y`/`n`
9. Proceed and generate? `y`/`n`

### Approach 2: expect script

For environments where InquirerPy is installed and arrow-key menus appear:

```bash
#!/usr/bin/env expect
spawn worldflux init ./demo-project --force
expect "Project name*"
send "demo-project\r"
expect "Environment type*"
# Press down once for mujoco, then Enter
send "\033\[B\r"
expect "Observation shape*"
send "\r"          ;# accept default 39
expect "Action dimension*"
send "\r"          ;# accept default 6
expect "Choose model*"
send "\r"          ;# accept recommended
expect "Total training steps*"
send "\r"          ;# accept default
expect "Batch size*"
send "\r"          ;# accept default
expect "Prefer GPU*"
send "n\r"
expect "Proceed*"
send "\r"          ;# accept default Yes
expect eof
```

### Approach 3: Uninstall InquirerPy temporarily

```bash
pip uninstall InquirerPy -y
# Now worldflux init falls back to Rich prompts (stdin-pipeable)
# ... run demo ...
pip install InquirerPy
```

## Recording Steps

### 1. Record the cast file

```bash
# 90-second version
chmod +x scripts/demo/record_demo.sh
./scripts/demo/record_demo.sh
# Output: worldflux-demo-90s.cast

# 30-second version
chmod +x scripts/demo/record_short.sh
./scripts/demo/record_short.sh
# Output: worldflux-demo-30s.cast
```

### 2. Preview the recording

```bash
asciinema play worldflux-demo-90s.cast
```

### 3. Convert to GIF (for GitHub README)

```bash
agg worldflux-demo-30s.cast worldflux-demo-30s.gif \
    --font-family "JetBrains Mono" \
    --font-size 18 \
    --theme asciinema \
    --cols 80 --rows 24 \
    --speed 1.0
```

### 4. Convert to MP4 (for YouTube / Twitter)

```bash
# Step 1: Cast -> GIF (high quality intermediate)
agg worldflux-demo-90s.cast worldflux-demo-90s-hq.gif \
    --font-family "JetBrains Mono" \
    --font-size 18 \
    --cols 100 --rows 30 \
    --speed 1.0

# Step 2: GIF -> MP4 (H.264 for broad compatibility)
ffmpeg -i worldflux-demo-90s-hq.gif \
    -movflags faststart \
    -pix_fmt yuv420p \
    -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=#1a1a2e" \
    -c:v libx264 -preset slow -crf 18 \
    worldflux-demo-90s.mp4
```

### 5. Alternative: Direct cast -> MP4 via svg-term + ffmpeg

```bash
# If agg quality is insufficient, use svg-term-cli for SVG frames
npx svg-term-cli --in worldflux-demo-90s.cast --out demo.svg \
    --window --no-cursor --width 100 --height 30

# Then rasterize with headless Chrome / Playwright if needed
```

## Post-Production

### Text Overlays (optional, via ffmpeg)

```bash
ffmpeg -i worldflux-demo-90s.mp4 \
    -vf "drawtext=text='pip install worldflux':fontcolor=white:fontsize=24:\
x=(w-text_w)/2:y=h-50:enable='between(t,85,90)'" \
    -c:v libx264 -preset slow -crf 18 \
    worldflux-demo-90s-final.mp4
```

### Adding Audio Narration (optional)

```bash
ffmpeg -i worldflux-demo-90s.mp4 -i narration.m4a \
    -c:v copy -c:a aac -shortest \
    worldflux-demo-90s-narrated.mp4
```

## Distribution Formats

| Format | Use Case | Target Size |
|--------|----------|-------------|
| `.cast` | asciinema.org embed, internal review | < 100 KB |
| `.gif` | GitHub README, social cards | < 5 MB |
| `.mp4` | YouTube, Twitter/X, blog posts | < 20 MB |

## Narration Script (for voiceover)

> WorldFlux is an open-source world-model reinforcement learning framework.
>
> Install it with pip. One command.
>
> Scaffold a new project with worldflux init. Choose your environment,
> observation shape, and model -- the CLI guides you through everything.
>
> The generated project includes train.py, inference.py, and a worldflux.toml
> configuration file. Ready to train out of the box.
>
> What makes WorldFlux unique is parity verification. We prove -- statistically
> -- that our implementations match upstream baselines. No silent regressions.
>
> Generate a parity badge for your README. Show your users the proof.
>
> WorldFlux. Open source. Parity-proven. Production-ready.
> Star us on GitHub.

## Checklist

- [ ] Terminal font set to JetBrains Mono 18px
- [ ] Terminal background set to `#1a1a2e`
- [ ] Terminal resolution 1920x1080
- [ ] Clean Python environment (no conflicting packages)
- [ ] asciinema installed and authenticated (`asciinema auth`)
- [ ] Test run without recording (`--no-rec` flag)
- [ ] Record 90-second demo
- [ ] Record 30-second short
- [ ] Preview both recordings with `asciinema play`
- [ ] Convert 30s to GIF for README
- [ ] Convert 90s to MP4 for YouTube/Twitter
- [ ] Upload cast files to asciinema.org (optional)
- [ ] Verify file sizes are within targets
