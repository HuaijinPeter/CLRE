# Figure assets

## `clre_pipeline.png` / `clre_pipeline.pdf`

The method overview figure used in the paper (Fig. 1, "Overview of CLRE").
Reproduced here unchanged — no redrawing, no relabelling, no change to the
scientific content.

| | |
| :--- | :--- |
| Source document | `ACC_rev0.tex` (the current manuscript revision) |
| LaTeX reference | `\includegraphics[width=\textwidth]{figures/Pipeline.pdf}` |
| Actual source file | `figures/pipeline.pdf` — the manuscript builds under MiKTeX/Windows, where the path resolves case-insensitively; no exact-case `Pipeline.pdf` exists in the manuscript tree |
| Source SHA-256 | `77a73282831c7e29e55334b13c5a2d0eb284dc6d6c32439ebd44eca358cc403a` |
| Original authoring tool | Inkscape 1.4.2 (vector, 2069 × 585 pt, 1 page) |
| `clre_pipeline.pdf` | byte-identical copy of that source, kept as the vector original |
| `clre_pipeline.png` | 2000 × 599 px, rendered for GitHub |

Two other pipeline figures exist in the manuscript tree and were **not** used:
`Pipeline3.pdf` (a restyled variant referenced by the superseded
`root.tex` + `sections/` split of the manuscript) and `pipeline2.pdf` (an
earlier, more illustrative layout that still carries superseded terminology
such as "Frozen VAD" and "Hard OBB Gate").

### How the PNG was produced

Rasterised at 4× the display width and downsampled, so the text stays crisp at
GitHub's rendering width:

```bash
# 1. rasterise the vector source at 288 dpi  ->  8276 x 2340 px
pdftoppm -png -r 288 -singlefile figures/pipeline.pdf hi_pipeline

# 2. crop surrounding whitespace, add a 1.2 % uniform margin, downsample to
#    2000 px wide with Lanczos resampling
python - <<'PY'
from PIL import Image, ImageChops
im = Image.open("hi_pipeline.png").convert("RGB")
bg = Image.new("RGB", im.size, (255, 255, 255))
core = im.crop(ImageChops.difference(im, bg).getbbox())
pad = round(core.size[0] * 0.012)
padded = Image.new("RGB", (core.size[0] + 2*pad, core.size[1] + 2*pad), (255, 255, 255))
padded.paste(core, (pad, pad))
h = round(padded.size[1] * 2000 / padded.size[0])
padded.resize((2000, h), Image.LANCZOS).save("clre_pipeline.png", "PNG", optimize=True)
PY
```

The source PDF is already tightly cropped, so step 2's crop is a no-op and only
the uniform margin is added. The README displays the PNG at `width="950"`.

### Updating it

If the figure changes in the manuscript, re-run the two commands above against
the new `figures/pipeline.pdf`, replace both files here, and update the
SHA-256 in the table. Keep the file names stable so the README does not need
editing.
