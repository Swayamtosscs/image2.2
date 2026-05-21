# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A single-file Streamlit app (`app.py`, ~760 lines) that generates AI fashion advertisement images for the brand **Hatchers Clothing Co.** Users upload clothing photos, the app sends them to Google Gemini's image generation with a detailed prompt, then composites branding overlays (logo, price badge, Instagram handle with icon) via SVG rendering.

## Running the App

```bash
source venv/bin/activate
streamlit run app.py
```

Requires a Google Gemini API key set via `GEMINI_API_KEY` in `.env` (auto-loaded via `python-dotenv`), or entered in the UI.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install streamlit pillow google-genai cairosvg python-dotenv
```

## Architecture

### Core Functions

1. **`generate_ad()`** — Sends up to 3 images to Gemini (style reference + clothing item + brand emblem) plus a heavily engineered prompt. Supports 3 model tiers via `MODEL_MAP`. Includes retry logic for 429 rate limits. Returns a raw PIL Image.

2. **`create_svg_overlay()`** — Builds an SVG overlay at full image resolution, renders it to PNG via `cairosvg`, and alpha-composites it onto the generated image. Elements:
   - White border frame
   - Top-right slanted polygon badge with model number + price
   - Bottom-left rotated vertical Instagram handle + Instagram icon (from `assets/insta.svg`)
   - Top-left brand logo (`assets/logo.png`) with adaptive colorization (dark on light bg, white on dark bg) — brightness sampled from the logo region

3. **`_colorize_alpha()`** — Recolors all non-transparent pixels to a given color while preserving alpha channel.

### Prompt Engineering (inside `generate_ad`)

The prompt has multiple conditional branches based on which images are available (reference, emblem, both, neither). Key sections appended to all variants:
- **Clothing detail** — strict instruction to reproduce ONLY what exists in the original clothing photo
- **Styling rules** — formal vs casual shirt handling, pant contrast
- **Pose rules** — no arm crossing, shirt must stay visible
- **Composition zones** — reserves top-left (logo), top-right (price), bottom-left (IG) as clear areas. Top-left must be uniform tone for logo readability
- **Emblem rules** — brand emblem carved into background scene, exact shape reproduction
- **Technical** — bright/fresh lighting, avoid dark/moody, aspect ratio

### Data Structures

- **`MODEL_MAP`** — Maps UI labels to Gemini model names, per-resolution costs, and capability flags
- **`FASHION_BACKGROUNDS`** (37 entries) — Scene descriptions categorized: Studio/Indoor, Urban/Street, Nature/Outdoor, Architectural/Luxury. All biased toward bright, well-lit settings per client feedback.
- **`FASHION_POSES`** (22 entries) — Pose descriptions categorized: Walking, Standing, Leaning, Seated, Candid, Dynamic angles, With accessories. All avoid obscuring the shirt front.

### Output Structure

Each generation batch saves to `output/<YYYYMMDD_HHMMSS>/`:
```
output/20260404_143025/
├── batch_info.txt                    # total cost, time, model used
└── <shirt_name>/
    ├── input/<original_image>        # uploaded clothing photo
    ├── generated/
    │   ├── ai_generated_1.jpg        # raw Gemini output
    │   └── final_ad_1.jpg            # with SVG overlay applied
    └── info.txt                      # per-campaign: cost, pose, background, model, time
```

## Assets

- `assets/logo.png` — Hatchers brand logo (used in SVG overlay, adaptively colorized)
- `assets/emblem.png` — Hatchers geometric emblem (sent to Gemini for background scene integration)
- `assets/insta.svg` — Instagram icon (rendered to PNG, colorized, embedded in SVG overlay)
- `reference.jpeg` — Style reference photo sent to Gemini (also at `assets/reference.jpeg`)

## Key Behaviors

- Logo adaptive color uses brightness threshold of 128 with an expanded sampling region around the logo area
- Logo position has 5% horizontal and 3% vertical margin from edges
- The prompt tells Gemini the top-left zone must be uniformly light OR dark (not mixed) so the overlay logo stays readable
- Emblem instruction tells Gemini to reproduce the EXACT shape from the image — no text description of the shape, to avoid Gemini reinterpreting it
- Clothing reproduction is strict: reproduce ONLY what exists in the original photo, do NOT add logos/symbols that aren't there
- Cost tracking converts USD to INR at 85x rate
- Session state (`st.session_state.generated_campaigns`) persists results across Streamlit reruns so download buttons work
