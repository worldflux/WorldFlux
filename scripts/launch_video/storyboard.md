# WorldFlux Teaser Video — Storyboard (28s)

> **Canvas**: 1080x1080 px | **BG**: #0D1117 | **FPS**: 30 | **Duration**: 28 sec
>
> All measurements in px from top-left origin. "cx" = center-x (540px).

---

## Design Tokens

```
COLORS:
  bg:          #0D1117    (GitHub Dark)
  text-primary:#E6EDF3    (body, code)
  text-muted:  #8B949E    (labels, secondary)
  accent-blue: #58A6FF    (highlight, links)
  accent-green:#7EE787    (success, checkmarks)
  accent-orange:#F0883E   (stats, attention)
  code-bg:     #161B22    (code block fill)
  strike-red:  #F85149    (strikethrough line)

FONTS:
  code:    JetBrains Mono, 28px (code blocks), 22px (terminal)
  heading: Inter Bold, 64px (hero), 48px (section), 36px (sub)
  stats:   Inter Black, 72px (big number), 32px (label)
  caption: Inter Regular, 24px (subtitles)

SPACING:
  margin:     80px (outer edge safe zone)
  code-pad:   40px (inside code blocks)
  line-height: 1.5 (code), 1.3 (headings)

SUBTITLE BAR:
  position: bottom 60px
  bg:       #0D1117 CC (80% opacity)
  height:   64px
  font:     Inter Regular 24px #E6EDF3
  align:    center
```

---

## Scene 1 — Hook (0:00–0:05)

### Phase 1A: "WEEKS" (0:00–0:03)

```
┌─────────────────────────────────────────┐
│                                         │
│                                         │
│                                         │
│                                         │
│             W E E K S                   │  ← Inter Black 72px #F0883E
│                                         │     center (cx, 480)
│                                         │     fade-in: 0.0s → 0.5s (opacity 0→1)
│                                         │
│                                         │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ World model integration takes   │    │  ← subtitle bar
│  │ weeks.                          │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

### Phase 1B: Strikethrough → "ONE COMMAND" (0:03–0:05)

```
┌─────────────────────────────────────────┐
│                                         │
│                                         │
│                                         │
│            ~~WEEKS~~                    │  ← strikethrough #F85149
│                                         │     fades/shrinks to y=380
│       O N E   C O M M A N D            │  ← Inter Black 72px #7EE787
│                                         │     center (cx, 540)
│                                         │     scale: 0.8→1.0 ease-out-back
│                                         │     + glow pulse (#7EE787 40%)
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ What if it was one command?     │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

**Animation Timeline**:
| Time | Event |
|------|-------|
| 0.0–0.5s | "WEEKS" fades in at center |
| 1.5s | VO hits "weeks" |
| 2.0–2.5s | Red line (#F85149, 4px) draws left→right through "WEEKS" |
| 2.5–3.0s | "WEEKS" shrinks to 40px, moves up to y=380, 50% opacity |
| 3.0–3.5s | "ONE COMMAND" scales in below (0.8→1.0, ease-out-back) |
| 3.5–4.0s | Green glow pulse (1 cycle) |
| 4.5–5.0s | Everything fades out |

**Production Notes**:
- "WEEKS" uses the abstracted form (not "2-6 WEEKS") — more impactful at scroll speed
- Strikethrough is at vertical midpoint of text (y = baseline - cap_height/2)
- The transition from "WEEKS" to "ONE COMMAND" is the key scroll-stop moment
- Must complete within first 3 seconds to hook viewers

**Transition to Scene 2**: Cross-dissolve 0.3s

---

## Scene 2 — Intro (0:05–0:07)

```
┌─────────────────────────────────────────┐
│                                         │
│                                         │
│                                         │
│                                         │
│              Y O S H I                  │  ← Inter Black 64px #E6EDF3
│                                         │     center (cx, 440)
│           Founder ┌────┐                │  ← Inter Regular 28px #8B949E
│                   │LOGO│                │     "Founder" left of logo
│                   └────┘                │     logo.svg 64x64
│                                         │     center (cx, 540)
│                                         │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ I'm Yoshi, founder of WorldFlux│    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

**Layout**:
- "YOSHI": Inter Black 64px #E6EDF3, center (cx, 440), letter-spacing +4px
- "Founder" + logo row: centered at (cx, 540)
  - "Founder": Inter Regular 28px #8B949E, right-aligned to cx-10
  - WorldFlux logo: 64x64, left edge at cx+10
- Minimal composition — name is the hero

**Animation Timeline**:
| Time | Event |
|------|-------|
| 5.0–5.3s | "YOSHI" fades in (opacity 0→1, 0.3s) with subtle scale (0.95→1.0) |
| 5.3–5.6s | "Founder" + logo row fades in below |
| 5.6–6.8s | Hold — let the name register |
| 6.8–7.0s | Fast fade out (0.2s) |

**Production Notes**:
- 2 seconds is tight — animations must be snappy
- Name in all-caps with letter-spacing gives gravitas
- Logo is small here (64px) — it's about the person, not the brand yet
- No SFX — just VO

**Transition to Scene 3**: Cross-dissolve 0.3s

---

## Scene 3 — Demo / HERO SHOT (0:07–0:15)

```
┌─────────────────────────────────────────┐
│  ┌─────────────────────────────────┐    │
│  │ ●  ●  ●   Terminal             │    │  ← terminal chrome bar
│  ├─────────────────────────────────┤    │     code-bg: #161B22
│  │                                 │    │     position: (60, 80) to (1020, 760)
│  │  $ worldflux init              │    │  ← JetBrains Mono 22px
│  │                                 │    │     $ in #7EE787, cmd in #E6EDF3
│  │  🚀 WorldFlux CLI              │    │
│  │                                 │    │
│  │  ┌─ Guided Setup ───────────┐  │    │  ← Rich panel, border #58A6FF
│  │  │ Project: my-world-model  │  │    │
│  │  │ Env: atari               │  │    │
│  │  │ Model: dreamer:ci [rec]  │  │    │     [rec] in #7EE787
│  │  │ Steps: 100K  Batch: 16   │  │    │
│  │  └──────────────────────────┘  │    │
│  │                                 │    │
│  │  ✅ Project created!            │    │  ← ✅ in #7EE787
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ One command — model, training,  │    │
│  │ inference — ready to run.       │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

**Terminal Chrome**:
- 3 dots: #F85149 (red), #F0883E (orange), #7EE787 (green) — 12px circles
- Tab title: "Terminal" in #8B949E 16px
- BG: #0D1117 with 1px border #30363D
- Border-radius: 12px top corners
- Inner padding: 32px

**This is a REAL screen recording** — not animated. Record with OBS, then speed up to fit.

**Animation Timeline**:
| Time | Event |
|------|-------|
| 7.0–7.3s | Terminal window fades in (empty dark shell) |
| 7.3–8.0s | `$ worldflux init` types in at natural speed |
| 8.0–8.3s | "🚀 WorldFlux CLI" banner appears |
| 8.3–11.0s | Wizard steps play at 4x speed (project name → env → model → config) |
| 11.0–12.0s | Configuration summary panel holds for 1s (readable) |
| 12.0–13.0s | "✅ Project created: ./my-world-model" appears with green flash |
| 13.0–14.0s | Next Steps panel flashes briefly |
| 14.0–15.0s | Hold on final state |

**Key Visual Moments** (must be readable even at 4x speed):
1. The Rich panels with blue borders (#58A6FF) — visually distinctive
2. Model recommendation `[recommended]` tag in green
3. The ✅ confirmation — the payoff moment
4. "Next Steps" showing `worldflux train` — implies the full workflow exists

**Recording Notes**:
- Run `worldflux init` for real in a terminal with dark theme (#0D1117 bg)
- Use default selections (atari → dreamer:ci → 100K steps) for a smooth path
- Record at 1x speed, then speed up in DaVinci Resolve to fit 8s
- The Rich UI formatting does the visual heavy lifting — no extra graphics needed

**Transition to Scene 4**: Cut (no dissolve — technical scenes stay crisp)

---

## Scene 4 — Architecture Switch (0:15–0:20)

```
┌─────────────────────────────────────────┐
│                                         │
│  ┌──────────────┐ ⟷ ┌──────────────┐   │
│  │  "dreamerv3"  │   │  "tdmpc2"    │   │  ← two code cards
│  │               │   │              │   │     left: (80, 200), right: (560, 200)
│  │  model =      │   │  model =     │   │     each: 420x300
│  │  create_world │   │  create_world│   │     string in #58A6FF
│  │  _model(      │   │  _model(     │   │
│  │  "dreamerv3", │   │  "tdmpc2",   │   │
│  │    ...)       │   │    ...)      │   │
│  └──────────────┘   └──────────────┘   │
│                                         │
│         ⟷  animated arrow (cx, 350)     │  ← #58A6FF, pulsing
│                                         │
│           S a m e   A P I .             │  ← Inter Bold 36px #8B949E
│                                         │     center (cx, 720)
│  ┌─────────────────────────────────┐    │
│  │ Swap architectures in one line. │    │
│  │ Same API.                       │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

**Animation Timeline**:
| Time | Event |
|------|-------|
| 15.0–15.4s | Left card slides in from left |
| 15.4–15.8s | Right card slides in from right |
| 15.8–16.3s | Arrow animates between cards (draw on) |
| 16.3–16.8s | Arrow pulses 2x in #58A6FF |
| 16.8–17.5s | "dreamerv3" text highlights (glow) |
| 17.5–18.2s | "tdmpc2" text highlights (glow) |
| 18.5–19.5s | "Same API." fades in at bottom |
| 19.5–20.0s | Hold |

**Production Notes**:
- Arrow: double-headed (⟷), line weight 3px, animated with dash offset
- Code cards: same `code-bg` #161B22, border 1px #30363D, border-radius 8px
- Only the string argument differs — visual reinforcement of "one line change"
- "Same API." is the key text — make it prominent

**Transition to Scene 5**: Cross-dissolve 0.5s

---

## Scene 5 — Close (0:20–0:28)

### Phase 5A: Tagline (0:20–0:24)

```
┌─────────────────────────────────────────┐
│                                         │
│                                         │
│                                         │
│              ┌────────┐                 │
│              │  LOGO  │                 │  ← logo.svg 160x160
│              └────────┘                 │     center (cx, 360)
│                                         │
│       One API. Infinite Imagination.    │  ← Inter Bold 40px #E6EDF3
│                                         │     center (cx, 560)
│                                         │     tracking: +2px
│                                         │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ One API. Infinite imagination.  │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

### Phase 5B: CTA (0:24–0:28)

```
┌─────────────────────────────────────────┐
│                                         │
│              ┌────────┐                 │
│              │  LOGO  │                 │  ← same position, stays
│              └────────┘                 │
│                                         │
│       One API. Infinite Imagination.    │  ← stays, dims to 60% opacity
│                                         │
│         ┌──────────────────┐            │
│         │   COMING SOON    │            │  ← badge: Inter Black 28px #0D1117
│         └──────────────────┘            │     bg: #58A6FF, border-radius 20px
│                                         │     center (cx, 660), pill shape
│  ┌────────────────┐ ┌────────────────┐  │
│  │  ✦ Follow      │ │  ✉ DM for     │  │  ← two CTA buttons
│  │  @[handle]     │ │  early access →│  │     left: (140, 740)-(500, 820)
│  └────────────────┘ └────────────────┘  │     right: (580, 740)-(940, 820)
│                                         │     border: 2px #58A6FF
│  ┌─────────────────────────────────┐    │
│  │ Coming soon. DM for early      │    │
│  │ access.                         │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

**"COMING SOON" Badge**:
- Shape: Pill/rounded rectangle (border-radius 20px)
- BG: #58A6FF (accent blue)
- Text: "COMING SOON" Inter Black 28px #0D1117 (dark on blue)
- Position: center (cx, 660)
- Width: auto (padding 32px horizontal, 12px vertical)
- Subtle pulse animation (scale 1.0→1.03→1.0, 2s cycle)

**CTA Buttons**:
- Left: X/Twitter icon + `Follow @[handle]`
  - Font: Inter Bold 22px #E6EDF3
  - Border: 2px solid #58A6FF
  - BG: transparent
- Right: Envelope icon + `DM for early access →`
  - Same style
  - Arrow (→) pulses gently

**Animation Timeline**:
| Time | Event |
|------|-------|
| 20.0–20.8s | Logo fades in with subtle scale (0.9→1.0) |
| 20.8–21.5s | "One API." fades in |
| 21.5–22.5s | "Infinite Imagination." fades in |
| 22.5–23.0s | Subtle ambient glow around logo (#58A6FF 15% opacity) |
| 23.5s | Tagline dims to 60% opacity |
| 23.5–24.3s | "COMING SOON" badge scales in (0.8→1.0) with bounce |
| 24.3–24.8s | Left CTA slides up from bottom |
| 24.8–25.3s | Right CTA slides up (0.5s stagger) |
| 25.3–26.5s | Badge pulse + arrow pulse |
| 26.5–28.0s | Hold — all elements visible, clean final frame |

**Production Notes**:
- Replace `[handle]` with your actual X username before production
- The final frame is also the thumbnail — all elements must read at small size
- "COMING SOON" badge is the visual anchor
- No GitHub URL — repo is private
- Consider holding final frame +1s for X auto-loop buffer
- CTA buttons: slight drop shadow (0 2px 8px rgba(0,0,0,0.3))

---

## Global Production Notes

### Subtitle Specifications
- **Font**: Inter Regular 24px
- **Color**: #E6EDF3
- **Background**: #0D1117 at 80% opacity (CC alpha)
- **Position**: bottom 60px from edge, centered
- **Max width**: 900px (margin 90px each side)
- **Max lines**: 2 per subtitle card
- **Timing**: sync to VO, appear 0.2s before speech, disappear 0.3s after

### Transition Inventory
| From → To | Type | Duration |
|-----------|------|----------|
| 1 → 2 | Cross-dissolve | 0.3s |
| 2 → 3 | Cross-dissolve | 0.3s |
| 3 → 4 | Cut | 0s |
| 4 → 5 | Cross-dissolve | 0.5s |

### Color Palette Quick Reference
```
#0D1117  ████  Background
#161B22  ████  Code block BG
#30363D  ████  Borders
#8B949E  ████  Muted text
#E6EDF3  ████  Primary text
#58A6FF  ████  Accent blue
#7EE787  ████  Success green
#F0883E  ████  Stats orange
#F85149  ████  Error red
```
