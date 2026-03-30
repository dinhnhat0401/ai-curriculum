# Building a Demo That Sells

**Time: ~20 hours | File: `streamlit_app.py`**

---

## What Makes a Great Demo

1. **Solves a validated problem** (from discovery, not imagination)
2. **Works on real data** (not toy examples)
3. **Shows measurable results** (accuracy, time saved, cost reduced)
4. **Is visually clear** (someone can understand without explanation)
5. **Handles edge cases** (doesn't break on the first unusual input)
6. **Has evaluation metrics built in** (numbers, not vibes)

## The Demo Script (5 minutes)

| Time | Section | What to Show |
|------|---------|-------------|
| 0:00 | The Problem | "Your team spends X hours/week on Y" |
| 0:30 | The Solution | "We built Z that automates Y" |
| 1:00 | Live Demo | Show it working on real data |
| 3:00 | Results | "X% accuracy, Y second response time, $Z cost per query" |
| 4:00 | Architecture | Brief technical overview (for technical audience) |
| 5:00 | Next Steps | "Here's what a pilot looks like" |

## Technical Stack for Demos

**Recommended:** Streamlit

Why: Python-native, fast to build, good enough for demos, no frontend expertise needed. You can go from zero to working demo in 2-3 hours.

When to upgrade: if you need custom styling, mobile support, or real-time collaboration, consider Next.js or Gradio.

## Demo Pitfalls

- Demoing only cherry-picked examples (always have a "try your own" option)
- Over-engineering the UI (substance over style)
- No error handling (when it fails in front of a client, fail gracefully)
- No metrics (claims without numbers don't convert)
- Not practicing the demo script (practice at least 3 times before showing anyone)
