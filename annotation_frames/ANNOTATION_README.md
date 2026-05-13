# Board Annotation Guide

Thank you for annotating these frames! Your markups will be used to calibrate the
board detection pipeline so it knows **exactly** where the dasher boards are in each
camera angle.

---

## What to annotate

Draw two colored lines across the boards in **every frame** you annotate:

| Color | What to draw | Notes |
|-------|-------------|-------|
| 🔴 **Red** | Top edge of boards | Where the transparent glass begins / the top of the white board face |
| 🟡 **Yellow** | Bottom edge of boards | Where the boards meet the ice (often there's a gold/yellow kickplate here) |

You only need to trace across boards that are **facing the camera** (the far-side boards
you'd replace an ad on). You do NOT need to mark boards that are nearly edge-on or the
near-side boards in tight sideline shots.

---

## Frame descriptions

| Filename | Camera angle | Notes |
|----------|-------------|-------|
| `v1_f010_overhead_end_zone.jpg` | Overhead end-zone, VGK arena | Far boards curve left-to-right |
| `v1_f317_overhead_end_zone.jpg` | Same angle, later in clip | Camera has panned slightly |
| `v1_f634_overhead_end_zone.jpg` | Same angle, end of clip | Far boards with more curve visible |
| `v3_f010_overhead_full.jpg` | High overhead, MTL arena | Wide view — mark both the near AND far boards if visible |
| `v3_f200_overhead_full.jpg` | Same, later | Check if camera angle changed |
| `v4_f010_overhead_near.jpg` | Tighter overhead, MTL arena | Near-side boards more prominent |

---

## How to annotate on Mac

**Option 1 — Preview (built-in, easiest):**
1. Open the image in Preview
2. Click the **Markup toolbar** button (pencil icon, top-right)
3. Select the **Draw** tool (squiggly line icon)
4. Pick your color (red or yellow) from the color swatch
5. Set line thickness to about **6–8 px**
6. Draw a line along the top of the boards (red), then the bottom (yellow)
7. Save as **new file**: e.g. `v1_f010_overhead_end_zone_annotated.jpg`

**Option 2 — Photoshop / Figma / Procreate:**
- Same idea: draw two lines in red and yellow

**Option 3 — Just a photo with your finger/stylus on screen:**
If easier, even a photo of the screen with lines drawn on it works!

---

## Example of what good annotation looks like

```
[Image]
  ←──── RED LINE ────────────────────────────────────→   ← top of boards (glass starts here)
  ←── YELLOW LINE ──────────────────────────────────→   ← bottom of boards (ice level)
         [board face between the two lines]
```

---

## Tips

- The boards are the **solid-colored panel** between the ice and the glass
- The **yellow/gold kickplate** at the very bottom of most boards is a great guide for the yellow line
- If the boards curve, follow the curve — don't just draw a straight horizontal line
- Don't worry about perfect pixel accuracy — within ~5px is plenty

---

## Save annotated files as

Same filename with `_annotated` appended:
- `v1_f010_overhead_end_zone_annotated.jpg` (or `.png`)

Drop them back in this folder when done.
