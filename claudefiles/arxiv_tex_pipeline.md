# Claudefile: arxiv LaTeX → Multimodal Training Corpus Pipeline

## Context for Implementing Claude

You're building infrastructure for a language model training project. The goal is to create a pipeline that converts arxiv papers (from LaTeX source) into a format suitable for training multimodal language models — and also supports the inverse direction (model outputs LaTeX → compiled PDF) for self-play and evaluation.

This is plumbing, not research. The person who wrote this spec has already talked through the design and believes it's straightforward. Your job is to make it work.

## What You're Building

### Forward Pipeline: arxiv → Training Data

**Input:** arxiv paper ID (e.g., `2306.13649`) or path to extracted source tarball

**Output:**
1. `structured.json` — parsed paper with:
   - `sections`: list of `{title, level, content}` where content is text with placeholder tokens for figures/equations
   - `figures`: list of `{id, caption, image_base64, original_path}`
   - `equations`: list of `{id, latex_source, image_base64, context}` (context = surrounding sentences)
   - `metadata`: title, authors, abstract, arxiv_id

2. `clean_source.tex` — the original LaTeX with figures replaced by `[FIGURE:id]` tokens and equations optionally replaced by `[EQUATION:id]` tokens (configurable — sometimes you want raw LaTeX equations inline)

3. `rendered.pdf` — the compiled PDF for reference/verification

### Inverse Pipeline: LaTeX → Compiled Output

**Input:** LaTeX string (potentially model-generated)

**Output:**
- `success`: boolean
- `pdf_base64`: the rendered PDF if successful, null otherwise
- `error_log`: compilation errors if failed
- `warnings`: list of non-fatal issues

This enables:
- Binary reward signal for self-play (did it compile?)
- Richer evaluation (does the output look reasonable?)
- Training data augmentation (model writes variations of papers)

## Implementation Notes

### arxiv Source Access

arxiv provides source tarballs at predictable URLs:
```
https://arxiv.org/e-print/{arxiv_id}
```

These are usually gzipped tarballs containing `.tex` files and figure assets (`.png`, `.pdf`, `.eps`, `.jpg`).

Some papers have multiple `.tex` files (main + supplements). Look for the one that contains `\documentclass` or `\begin{document}`.

### LaTeX Parsing

You don't need a full LaTeX parser. The structure you care about is:

```
\section{...}
\subsection{...}
\begin{figure}...\includegraphics{...}...\caption{...}\end{figure}
\begin{equation}...\end{equation}
$...$  (inline math)
$$...$$ (display math)
\[...\]  (display math alternate)
```

Regex or simple state-machine parsing is fine. Libraries like `plasTeX` or `pylatexenc` exist but may be overkill.

Handle `\input{...}` and `\include{...}` by inlining the referenced files.

### Figure Extraction

Figures come in three forms:
1. **Referenced image files** — `\includegraphics{figures/foo.png}` — just read the file from the tarball
2. **PDF figures** — same, but you'll want to convert to PNG for training
3. **TikZ/PGF diagrams** — these are LaTeX code that compiles to figures. To extract: isolate the tikzpicture environment, compile it standalone, capture the output.

For (3), you can create a minimal document:
```latex
\documentclass[tikz]{standalone}
\begin{document}
<tikzpicture content here>
\end{document}
```
Then compile and convert to PNG.

### Equation Rendering

For equations you want as images:
```latex
\documentclass[preview,border=2pt]{standalone}
\usepackage{amsmath,amssymb}
\begin{document}
$<equation here>$
\end{document}
```
Compile with `pdflatex`, convert to PNG with `pdftoppm` or `pdf2image`.

### Compilation

Use `pdflatex` or `latexmk`. Run in a temp directory. Capture stdout/stderr for error reporting.

Many arxiv papers require multiple passes (for references, etc.). `latexmk` handles this automatically.

Some papers need specific packages. You'll want a fairly complete TeX installation (TeX Live full).

### Error Handling

arxiv papers have a long tail of weird setups:
- Custom style files (`.sty`) included in the tarball
- Missing fonts (usually fine to ignore)
- Packages that don't exist anymore
- Broken LaTeX that somehow compiled in 2015

For the forward pipeline: best-effort extraction. If a figure fails, log it and continue. If the whole paper fails to parse, return what you got.

For the inverse pipeline: strict. Compilation either works or it doesn't.

## Interface

Implement as a Python module with CLI:

```bash
# Forward pipeline
python -m arxiv_pipeline extract 2306.13649 --output-dir ./output/

# Inverse pipeline  
python -m arxiv_pipeline compile input.tex --output-dir ./output/

# Batch processing
python -m arxiv_pipeline batch-extract arxiv_ids.txt --output-dir ./corpus/
```

And as importable functions:

```python
from arxiv_pipeline import extract_paper, compile_latex

result = extract_paper("2306.13649")
# result.structured -> dict
# result.clean_source -> str
# result.rendered_pdf -> bytes

compile_result = compile_latex(tex_string)
# compile_result.success -> bool
# compile_result.pdf -> bytes | None
# compile_result.errors -> list[str]
```

## Dependencies

```
requests          # arxiv downloads
pylatexenc        # optional, for LaTeX parsing utilities
pdf2image         # PDF to PNG conversion
Pillow            # image handling
```

System dependencies:
- `texlive-full` or equivalent (for pdflatex, latexmk)
- `poppler-utils` (for pdftoppm)

## Success Criteria

1. Successfully extracts 90%+ of arxiv papers in a random sample of 100 CS papers
2. Figure extraction works for simple `\includegraphics` cases
3. Equation rendering produces legible images
4. Inverse pipeline correctly reports success/failure for valid/invalid LaTeX
5. Handles the GKD paper (2306.13649) as a test case

## Non-Goals (For Now)

- Perfect parsing of every LaTeX edge case
- Bibliography extraction (`.bib` files)
- Cross-reference resolution (theorem numbers, etc.)
- Handling Word docs or other non-LaTeX sources

## Test Cases

Use these arxiv IDs for testing:
- `2306.13649` — GKD paper, standard NeurIPS format
- `2312.06585` — recent ML paper with figures
- `1706.03762` — Attention Is All You Need, classic, heavy on equations

## Questions You Might Have

**Q: Should I preserve the exact LaTeX or normalize it?**
A: Preserve. The model should see real LaTeX as it appears in the wild.

**Q: What resolution for figure/equation images?**
A: 300 DPI is fine. We can always re-render later.

**Q: How to handle multi-file papers?**
A: Find the main `.tex` file (the one with `\documentclass`), inline all `\input`/`\include` references.

**Q: What if a paper has no source (PDF only)?**
A: Out of scope for this pipeline. Those need OCR, which is a different tool.

---

When you're done, the output should be a working Python package that another script can call to build a training corpus from a list of arxiv IDs.
