# WorldFlux Teaser Video — Production Guide (28s)

> 無料ツールのみで制作する実践ガイド。このドキュメントは制作担当者（自分）向け。

---

## Table of Contents

1. [Tool Setup](#1-tool-setup)
2. [Asset Preparation](#2-asset-preparation)
3. [VO Recording (OBS)](#3-vo-recording-obs)
4. [Screen Recording (OBS)](#4-screen-recording-obs)
5. [Animation (Motion Canvas)](#5-animation-motion-canvas)
6. [Editing & Compositing (DaVinci Resolve)](#6-editing--compositing-davinci-resolve)
7. [Subtitles](#7-subtitles)
8. [Audio Mix](#8-audio-mix)
9. [Export & Upload](#9-export--upload)
10. [Pre-Flight Checklist](#10-pre-flight-checklist)

---

## 1. Tool Setup

### Required (All Free)

| Tool | Version | Purpose | Install |
|------|---------|---------|---------|
| **DaVinci Resolve** | 19 Free | Edit, color, compositing, export | [blackmagicdesign.com](https://www.blackmagicdesign.com/products/davinciresolve/) |
| **OBS Studio** | 31+ | VO recording + screen capture | [obsproject.com](https://obsproject.com/) |
| **Motion Canvas** | latest | Programmatic animations | `npm init @motion-canvas` |
| **JetBrains Mono** | latest | Code font | [jetbrains.com/mono](https://www.jetbrains.com/mono/) |
| **Inter** | latest | UI font | [rsms.me/inter](https://rsms.me/inter/) |

### Optional

| Tool | Purpose |
|------|---------|
| Audacity | VO cleanup if OBS audio isn't clean enough |
| GIMP/Figma | Static frame mockups before animation |
| ffmpeg | Quick format conversion, thumbnail extraction |

### Font Installation

1. Download JetBrains Mono & Inter
2. Install system-wide (macOS: double-click .ttf → Install Font)
3. Restart DaVinci Resolve and Motion Canvas after installation
4. Verify: in DaVinci Resolve → Text+ → Font dropdown → search "JetBrains" and "Inter"

---

## 2. Asset Preparation

### Collect Existing Assets

```bash
# From project root
cp docs/assets/logo.svg scripts/launch_video/assets/
cp docs/assets/logo_transparent.svg scripts/launch_video/assets/
```

### Create Assets Directory

```bash
mkdir -p scripts/launch_video/assets/{raw,rendered,audio}
```

**Directory structure**:
```
scripts/launch_video/
├── script.md
├── storyboard.md
├── production_guide.md    ← this file
└── assets/
    ├── raw/               ← source files, OBS recordings
    ├── rendered/          ← Motion Canvas exports
    └── audio/             ← VO takes, BGM
```

### Logo Preparation

```bash
# SVG → high-res PNG for video (3x for 1080p)
# Option 1: Using Inkscape (free)
inkscape docs/assets/logo.svg \
  --export-type=png \
  --export-width=480 \
  --export-filename=scripts/launch_video/assets/logo_480.png

# Option 2: Using rsvg-convert (brew install librsvg)
rsvg-convert -w 480 docs/assets/logo.svg \
  > scripts/launch_video/assets/logo_480.png
```

---

## 3. VO Recording (OBS)

### OBS Audio Setup

1. **Sources** → Add → **Audio Input Capture** → select your mic
2. **Settings** → Audio:
   - Sample Rate: **48 kHz**
   - Channels: **Mono**
3. **Settings** → Output → Recording:
   - Type: Standard
   - Format: **mkv** (safer, convert to wav after)
   - Audio Encoder: AAC 192kbps
4. **Audio Mixer**: Set mic volume so peaks hit -6dB to -3dB (green/yellow, never red)

### Recording Process

1. **Environment**: Quiet room, close doors/windows. Closet with clothes = poor man's vocal booth.
2. **Mic Position**: 6-8 inches from mouth, slightly off-axis (reduce plosives)
3. **Pop Filter**: Use one if available. Tissue over a wire hanger works in a pinch.
4. Record all 5 scenes in one take (28s is short enough), then do 3 full takes:
   - `vo_full_take1.mkv`, `vo_full_take2.mkv`, `vo_full_take3.mkv`
5. **Between takes**: Clap once (creates a visible spike for sync)
6. **Pacing reference**: Print the clean copy (51 words), read along at ~150 WPM

### Post-Recording

```bash
# Convert MKV → WAV for editing
ffmpeg -i assets/raw/vo_full_take1.mkv \
  -vn -acodec pcm_s16le \
  assets/audio/vo_full_take1.wav
```

### VO Cleanup (if needed, in Audacity)

1. Open WAV → Effect → Noise Reduction → Get Noise Profile (select silent section) → Apply
2. Effect → Compressor: Threshold -18dB, Ratio 3:1, Attack 10ms
3. Effect → Normalize: -3dB peak
4. Export as WAV 48kHz 16-bit

---

## 4. Screen Recording (OBS)

### What to Record

Scene 3 (`worldflux init` demo) needs a real terminal screen recording. All other scenes are animated in Motion Canvas.

### OBS Canvas Setup for Init Demo

1. **Settings** → Video:
   - Base Resolution: **1080x1080**
   - Output Resolution: **1080x1080**
   - FPS: **30**
2. **Scene Setup**:
   - Add **Window Capture** of your terminal app
   - Crop/resize to fill 1080x1080
3. **Terminal Setup**:
   - App: iTerm2, Kitty, or Warp (any with rich rendering support)
   - Theme: Dark with BG **#0D1117** (match video BG exactly)
   - Font: JetBrains Mono 16pt
   - Window size: ~100 columns × 30 rows (fits the Rich panels well)
   - Hide tab bar, title bar if possible — just the terminal content

### Recording the Init Demo

1. **Pre-run**: Execute `worldflux init --force /tmp/test-run` once to ensure dependencies are cached and there are no first-run delays
2. **Prepare output directory**: `rm -rf /tmp/demo-project`
3. Start OBS recording
4. Type `worldflux init /tmp/demo-project` at natural speed
5. Select defaults at each prompt:
   - Project name: `my-world-model` (default)
   - Environment: `atari` (default)
   - Model: `dreamer:ci` (recommended, default)
   - Steps: `100,000` (recommended)
   - Batch: `16` (recommended)
   - Device: `gpu` or `cpu` (auto-detected)
6. Confirm when prompted
7. Wait 2s on the "✅ Project created" + "Next Steps" output
8. Stop recording

**Post-processing in DaVinci Resolve**:
- Import the full recording (~30-60s real-time)
- Speed ramp: keep `$ worldflux init` typing at 1x speed (natural feel)
- Speed up the wizard prompts to **4x** (Rich UI panels flash by impressively)
- Return to 1x for "✅ Project created" moment (let it land)
- Trim total to **8 seconds**
- Add terminal chrome overlay on V3 if the actual terminal title bar doesn't match the design spec

**Tip**: Record 3 takes. Pick the one where Rich panels render cleanest and the timing feels best.

---

## 5. Animation (Motion Canvas)

Motion Canvas produces programmatic, frame-perfect animations exported as image sequences.

### Project Setup

```bash
cd scripts/launch_video
npm init @motion-canvas
# Project name: worldflux-launch
cd worldflux-launch
npm install
```

### Key Scenes to Animate

| Scene | Priority | Complexity |
|-------|----------|------------|
| 1: "WEEKS" + strikethrough → "ONE COMMAND" | High | Medium |
| 2: Founder name + logo | High | Low |
| 3: Init demo — **OBS screen recording** (not animated) | — | — |
| 4: Split-screen code cards | High | Medium |
| 5A: Logo + tagline | High | Low |
| 5B: "COMING SOON" badge + CTA | High | Low |

Only **4 scenes** to animate (down from 8 in the 57s version).

### Example: Scene 1 Animation (Motion Canvas)

```typescript
// src/scenes/scene1.tsx
import { makeScene2D, Txt, Line } from '@motion-canvas/2d';
import { all, waitFor, createRef } from '@motion-canvas/core';

export default makeScene2D(function* (view) {
  const weeks = createRef<Txt>();
  const strike = createRef<Line>();
  const oneCmd = createRef<Txt>();

  view.fill('#0D1117');

  // "WEEKS" text
  view.add(
    <Txt
      ref={weeks}
      text="W E E K S"
      fontFamily="Inter"
      fontWeight={900}
      fontSize={72}
      fill="#F0883E"
      opacity={0}
      x={0} y={-60}
    />
  );

  // Strikethrough line
  view.add(
    <Line
      ref={strike}
      points={[[-200, -60], [200, -60]]}
      stroke="#F85149"
      lineWidth={4}
      end={0}
      opacity={0}
    />
  );

  // "ONE COMMAND" text
  view.add(
    <Txt
      ref={oneCmd}
      text="O N E   C O M M A N D"
      fontFamily="Inter"
      fontWeight={900}
      fontSize={72}
      fill="#7EE787"
      opacity={0}
      scale={0.8}
      x={0} y={60}
    />
  );

  // Fade in "WEEKS"
  yield* weeks().opacity(1, 0.5);
  yield* waitFor(1.5);

  // Draw strikethrough
  yield* strike().opacity(1, 0);
  yield* strike().end(1, 0.5);
  yield* waitFor(0.3);

  // Shrink "WEEKS", reveal "ONE COMMAND"
  yield* all(
    weeks().fontSize(40, 0.4),
    weeks().y(-120, 0.4),
    weeks().opacity(0.5, 0.4),
  );
  yield* all(
    oneCmd().opacity(1, 0.5),
    oneCmd().scale(1, 0.5),
  );

  // Green glow pulse (visual only in post)
  yield* waitFor(1.0);

  // Fade out everything
  yield* all(
    weeks().opacity(0, 0.4),
    strike().opacity(0, 0.4),
    oneCmd().opacity(0, 0.4),
  );
});
```

### Export Settings (Motion Canvas)

```typescript
// In project settings or vite.config.ts
export default {
  output: './output',
  size: { width: 1080, height: 1080 },
  fps: 30,
};
```

Export produces a numbered PNG image sequence. Import into DaVinci Resolve as clip.

### Import Image Sequence in DaVinci Resolve

1. Media Pool → Import → navigate to `output/` folder
2. Select first frame → Check "Image Sequence" option at bottom
3. Set frame rate to 30fps
4. Drag to timeline

---

## 6. Editing & Compositing (DaVinci Resolve)

### Project Settings

1. **File → Project Settings**:
   - Timeline Resolution: **1080 × 1080** (Custom)
   - Timeline Frame Rate: **30**
   - Playback Frame Rate: **30**
   - Color Science: DaVinci YRGB
   - Timeline Format: ProRes 422 (for editing, export as H.264 later)

### Timeline Structure

```
Video Track 4:  [Subtitle overlays — Text+]
Video Track 3:  [Overlay text, code highlights]
Video Track 2:  [Motion Canvas animations]
Video Track 1:  [Background fills, screen recording]
Audio Track 1:  [VO (selected best take, trimmed)]
Audio Track 2:  [BGM (lo-fi ambient, ducked)]
Audio Track 3:  [SFX — optional whoosh, dings]
```

### Assembly Order

1. **Import all assets**: VO WAVs, Motion Canvas sequences, init demo recording, logo PNG
2. **Lay down VO first**: Place best take on A1, trim silence, align to timecodes from script
3. **Place Scene 3 (init demo)**: Speed-ramped OBS recording on V1, terminal chrome overlay on V3 if needed
4. **Place Motion Canvas clips**: Align to VO timing on V2
5. **Add transitions**: Dissolves per storyboard transition inventory
6. **Add subtitles**: Text+ on V4 (see Section 7)
7. **Color pass**: Ensure all backgrounds match #0D1117 exactly
8. **Audio mix**: See Section 8

### Key DaVinci Resolve Techniques

**Speed ramp for init demo (Scene 3)**:
- Import full OBS recording to timeline
- Right-click → Retime Controls
- Set typing segment (first 1s) to 100% speed
- Set wizard segment to 400% speed (4x)
- Set "✅ Project created" segment back to 100% speed
- Trim entire clip to 8s total

**Background color fill**:
- Add a Solid Color generator (Effects → Generators → Solid Color)
- Set to #0D1117
- Place on V1 for the full timeline duration

---

## 7. Subtitles

### Creating Subtitles in DaVinci Resolve

**Method: Text+ on Track V4** (hardcoded subtitles)

1. Effects Library → Titles → Text+
2. Drag to V4 at each scene's start time
3. Configure for each card:
   - **Font**: Inter Regular
   - **Size**: 24px (0.0222 in Resolve units at 1080px)
   - **Color**: #E6EDF3
   - **Background**: Enable → #0D1117 at 80% opacity
   - **Position**: Y = -0.44 (bottom 60px equivalent in normalized coords)
   - **Alignment**: Center
4. Trim each Text+ clip to match VO timing (appear 0.2s before speech, disappear 0.3s after)

### Subtitle Content per Scene

| Scene | Start | End | Text |
|-------|-------|-----|------|
| 1-hook | 0:00.0 | 0:02.5 | World model integration takes weeks. |
| 1-flip | 0:02.8 | 0:04.8 | What if it was one command? |
| 2 | 0:05.0 | 0:06.8 | I'm Yoshi, founder of WorldFlux. |
| 3 | 0:07.2 | 0:14.5 | One command — model, training, inference — ready to run. |
| 4 | 0:15.2 | 0:19.5 | Swap architectures in one line. Same API. |
| 5-tag | 0:20.2 | 0:23.0 | One API. Infinite imagination. |
| 5-cta | 0:23.5 | 0:27.0 | Coming soon. DM for early access. |

**Note**: 7 subtitle cards total (down from 13 in the 57s version). All fit within 2-line / ~60 character limit.

---

## 8. Audio Mix

### VO Processing in DaVinci Resolve (Fairlight Page)

1. Select VO track → Inspector → EQ:
   - High-pass filter at 80Hz (remove rumble)
   - Gentle boost at 3kHz (+2dB, Q=1.0) for presence
2. Compressor:
   - Threshold: -18dB
   - Ratio: 3:1
   - Attack: 10ms
   - Release: 100ms
3. Normalize peaks to -6dB

### BGM Selection

**Recommended free sources** (royalty-free, CC-licensed):

| Source | URL | Style |
|--------|-----|-------|
| **Pixabay Music** | pixabay.com/music | Lo-fi, ambient, tech |
| **Free Music Archive** | freemusicarchive.org | Wide selection, check license |
| **Uppbeat** | uppbeat.io | Free tier with attribution |

**What to look for**:
- Genre: Lo-fi ambient / minimal electronic / tech background
- Tempo: 70-90 BPM (slow, unobtrusive)
- Key: Minor key preferred (serious, technical feel)
- Duration: 30+ seconds (loop-friendly; shorter than before since video is 28s)
- No vocals, no prominent melody

### BGM Mix Settings

1. Import BGM to Audio Track 2
2. Volume: **-18dB to -22dB** (well below VO)
3. **Ducking**: Use Fairlight's ducking feature or manual keyframes:
   - When VO is active: BGM at -22dB
   - During pauses (transitions): BGM at -16dB
4. Fade in: 0.5s at start (0:00)
5. Fade out: 1.5s at end (0:26.5–0:28)

### SFX (Optional)

| Sound | When | Source |
|-------|------|--------|
| Whoosh | Scene 1 strikethrough (2.0s) | freesound.org, search "whoosh subtle" |

SFX volume: -12dB, keep subtle. Only 1 SFX moment (down from 2 in the 57s version).

---

## 9. Export & Upload

### DaVinci Resolve Export Settings

**Deliver Page** settings:

| Setting | Value |
|---------|-------|
| Format | MP4 |
| Codec | H.264 |
| Resolution | 1080 × 1080 |
| Frame Rate | 30 |
| Quality | **Restrict to 2 Mbps** (28s × 2Mbps ≈ 7MB, under 8MB target) |
| Profile | High |
| Key Frames | Auto |
| Audio Codec | AAC |
| Audio Bitrate | 192 kbps |
| Data Levels | Auto |

**Filename**: `worldflux_teaser_28s_v1_1080x1080.mp4`

### File Size Verification

```bash
# Check file size (must be <8MB target, <15MB hard limit for free X)
ls -lh worldflux_teaser_28s_v1_1080x1080.mp4

# Target: ~7MB at 2 Mbps for 28 seconds
# If over 8MB, re-export with lower bitrate (try 1.5 Mbps)
# Or use ffmpeg to re-encode:
ffmpeg -i worldflux_teaser_28s_v1_1080x1080.mp4 \
  -c:v libx264 -b:v 1.5M -maxrate 2M -bufsize 4M \
  -c:a aac -b:a 192k \
  -movflags +faststart \
  worldflux_teaser_28s_v1_compressed.mp4
```

### Thumbnail Extraction

```bash
# Extract the final frame (CTA scene) for X thumbnail
ffmpeg -sseof -1 -i worldflux_teaser_28s_v1_1080x1080.mp4 \
  -frames:v 1 -update 1 \
  worldflux_teaser_thumbnail.png
```

### X (Twitter) Upload Specifications

| Requirement | Our Video | Status |
|-------------|-----------|--------|
| Max duration | 2:20 | 0:28 OK |
| Max file size | 512MB (Blue), 15MB (free) | Target <8MB OK |
| Aspect ratio | 1:1 supported | 1:1 OK |
| Resolution | Up to 1920x1200 | 1080x1080 OK |
| Codec | H.264 + AAC | H.264 + AAC OK |
| Frame rate | Up to 60fps | 30fps OK |

**Important**: 28秒 × 2Mbps ≈ 7MBなのでfree Xの15MB制限は余裕でクリア。ビットレートを上げたい場合は4Mbps（≈14MB）まで安全。

### Upload Procedure

1. X → New Post → Add Media → Select MP4
2. Wait for processing (1-3 min)
3. Preview the video — check:
   - Subtitles readable at mobile size
   - First frame is compelling (autoplay thumbnail)
   - Audio levels correct
4. Post text (suggested):

```
Every world model has a different API.
Integration takes weeks.

What if it was one command?

WorldFlux — One API. Infinite Imagination.

Coming soon. DM me for early access.

↓ 30-sec teaser ↓
```

5. **Pin the post** to your profile

**Note**: `pip install worldflux` は公開後に追加。Coming soon段階ではインストールコマンドを出さない。

---

## 10. Pre-Flight Checklist

### Technical Quality

- [ ] Resolution is exactly 1080x1080
- [ ] Frame rate is exactly 30fps
- [ ] File size is under 8MB (target) / 15MB (hard limit for free X)
- [ ] H.264 video + AAC audio codec
- [ ] No audio clipping (peaks below -3dB)
- [ ] BGM doesn't overpower VO
- [ ] All fonts render correctly (no fallback to system fonts)

### Content Quality

- [ ] Mute test: watch with sound off — content 100% understandable from subtitles + visuals
- [ ] First 3 seconds test: "WEEKS" → strikethrough completes and stops scrolling
- [ ] Technical accuracy: `worldflux init` recording is real (not mocked)
- [ ] Architecture switch shows correct API (`create_world_model`)
- [ ] No typos in overlay text or subtitles
- [ ] No GitHub URL or `pip install` command visible (repo is private)
- [ ] CTA is clear: "COMING SOON" + Follow + DM for early access all visible in final frame

### Visual Consistency

- [ ] Background color is #0D1117 throughout (no color shifts at transitions)
- [ ] Code font is JetBrains Mono everywhere
- [ ] Heading font is Inter everywhere
- [ ] Accent colors match design tokens (blue #58A6FF, green #7EE787, orange #F0883E)
- [ ] Subtitles are readable at mobile phone size (~375px viewport)
- [ ] Logo is sharp (not pixelated)

### Timing

- [ ] Total duration: 27-28 seconds
- [ ] VO timing matches visual transitions (51 words / ~150 WPM)
- [ ] Subtitles appear before VO, disappear after
- [ ] No scene feels rushed or too slow
- [ ] Pauses feel natural
- [ ] Read VO aloud — fits within 28 seconds with room for pauses

### X-Specific

- [ ] Video looks good in X's square preview
- [ ] First frame (auto-thumbnail) is visually interesting
- [ ] Final frame shows all CTA elements at small size
- [ ] Post text complements (doesn't repeat) video content
- [ ] Post is pinned to profile

---

## Production Timeline Estimate

| Phase | Tasks | Duration |
|-------|-------|----------|
| **Prep** | Install tools, collect assets, convert formats | 1 hour |
| **VO** | 3 full takes, select best, cleanup | 1 hour |
| **Animation** | Motion Canvas scenes (4 scenes, down from 8) | 2-3 hours |
| **Edit** | Assembly, timing, transitions in DaVinci | 1-2 hours |
| **Polish** | Subtitles, audio mix, color check | 1 hour |
| **QA** | Checklist, test export, get feedback | 30 min |
| **Total** | | **6-8 hours** |

**Tip**: Scene 2 (founder intro) and Scene 5A (logo + tagline) are the simplest animations — start there for quick wins. The init demo recording (Scene 3) is independent and can be done in parallel.
