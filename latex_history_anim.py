#!/usr/bin/env python3
"""
latex_history_anim.py

Usage:
    python latex_history_anim.py /path/to/repo --tex main.tex --out history.gif

What it does:
    - Finds all git commits that touched the specified .tex file
    - For each commit (oldest -> newest):
        - checks out the commit (detached HEAD)
        - builds the specified .tex into a PDF (tries latexmk, falls back to pdflatex)
        - uses pdftoppm to render up to `--max-pages` PNG pages
        - composes the PNG pages side-by-side into a single image (up to max-pages)
        - writes a composed PNG per commit
    - After all commits, composes a GIF (or AVI if you prefer) from the composed PNGs.

Note: The script leaves the repository checked out to the original branch at the end (attempts to).
"""

import argparse
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import imageio
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run(cmd, cwd=None, check=True, capture_output=False):
    logging.debug("RUN: %s (cwd=%s)", " ".join(cmd), cwd)
    return subprocess.run(cmd, cwd=cwd, check=check, stdout=(subprocess.PIPE if capture_output else None),
                          stderr=(subprocess.PIPE if capture_output else None), text=True)


def ensure_tools_exist():
    missing = []
    for tool in ("git", "pdftoppm"):
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        raise RuntimeError(f"Missing required tools in PATH: {', '.join(missing)}. Install them and retry.")


def get_commits_touching_file(repo_path, files):
    # returns commit hashes (oldest -> newest)
    cmd = ["git", "-C", str(repo_path), "log", "--pretty=format:%H", "--reverse", "--"] + files
    proc = run(cmd, capture_output=True)
    hashes = [h.strip() for h in proc.stdout.splitlines() if h.strip()]
    logging.info("Found %d commits touching %s", len(hashes), files)
    return hashes


def build_latex(repo_path, tex_path, workdir, build_outdir):
    """
    Try latexmk first. If it fails or missing, fall back to pdflatex (2 runs).
    tex_path must be a Path relative to repo_path (or absolute).
    The PDF should appear in build_outdir (if pdflatex used, use -output-directory).
    Returns path to the resulting PDF or raises RuntimeError on failure.
    """
    pdf_name = tex_path.with_suffix(".pdf").name
    build_outdir = Path(build_outdir)

    # Try latexmk
    latexmk_exe = shutil.which("latexmk")
    if latexmk_exe:
        cmd = [latexmk_exe, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-silent",
               "-jobname=" + tex_path.stem, str(tex_path)]
        # latexmk may not support -outdir consistently across all distributions; use WORKDIR technique:
        # run latexmk from the directory where tex is located, but set environment OUTDIR by tex engine options is tricky.
        # Simpler: run latexmk in a temp build folder with a copy of the tex and inputs.
        try:
            logging.info("Building with latexmk")
            run(cmd, cwd=workdir)
            produced_pdf = workdir / pdf_name
            if produced_pdf.exists():
                return produced_pdf
            # else fallback
            logging.warning("latexmk finished but PDF not found at %s", produced_pdf)
        except subprocess.CalledProcessError as e:
            logging.warning("latexmk failed: %s", getattr(e, "stderr", str(e)))

    # Fallback to pdflatex (2 passes)
    pdflatex_exe = shutil.which("pdflatex")
    if pdflatex_exe is None:
        raise RuntimeError("Neither latexmk nor pdflatex is available to build the document.")

    logging.info("Building with pdflatex (fallback)")
    # Ensure output directory exists and run pdflatex there with -output-directory
    build_outdir.mkdir(parents=True, exist_ok=True)
    # run pdflatex twice
    for i in range(2):
        cmd = [pdflatex_exe, "-interaction=nonstopmode", "-halt-on-error",
               "-output-directory", str(build_outdir), str(tex_path)]
        try:
            run(cmd, cwd=workdir)
        except subprocess.CalledProcessError as e:
            # capture output if available
            raise RuntimeError(f"pdflatex failed on pass {i+1}:\n{getattr(e, 'stderr', '') or e}")

    produced_pdf = build_outdir / pdf_name
    if not produced_pdf.exists():
        raise RuntimeError(f"Build succeeded but PDF not found at {produced_pdf}")
    return produced_pdf


def pdf_to_png_pages(pdf_path, out_prefix, dpi=150, max_pages=10):
    """
    Uses pdftoppm to create PNG pages.
    Returns list of created png paths (ordered page1..pagen), capped at max_pages.
    """
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise RuntimeError("pdftoppm not found (required). Install Poppler utilities.")

    # pdftoppm -png -r DPI input.pdf outprefix
    cmd = [pdftoppm, "-png", "-r", str(dpi), str(pdf_path), str(out_prefix)]
    run(cmd)
    # get number of digits for first page
    digits = 0
    for i in range(1, 3):
        if Path(str(out_prefix) + "-" + ("0" * i) + "1.png").exists():
            digits = i
            break

    # produced files like outprefix-1.png outprefix-2.png ...
    produced = []
    for i in range(1, max_pages + 1):
        p = Path(str(out_prefix) + "-" + ("0" * digits) + str(i) + ".png")
        if p.exists():
            produced.append(p)
        else:
            break
    return produced


def compose_side_by_side(images, max_pages=10, max_height=1200, gap=10):
    """
    Compose up to max_pages images side-by-side (horizontally).
    - images: list of PIL.Image paths or Image objects
    - max_height: target maximum height for composed image; each image is scaled to fit this height preserving aspect ratio.
    - gap: pixels between pages
    Returns a PIL.Image.
    """
    imgs = []
    for im in images[:max_pages]:
        if isinstance(im, (str, Path)):
            im = Image.open(im)
        imgs.append(im.convert("RGBA"))

    if not imgs:
        raise ValueError("No images to compose.")

    # scale each so that height <= max_height (maintain relative heights)
    scaled = []
    for im in imgs:
        w, h = im.size
        if h > max_height:
            scale = max_height / float(h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            im = im.resize((new_w, new_h), Image.LANCZOS)
        scaled.append(im)

    total_w = sum(im.width for im in scaled) + gap * (len(scaled) - 1)
    max_h = max(im.height for im in scaled)

    composed = Image.new("RGBA", (total_w, max_h), (255, 255, 255, 255))
    x = 0
    for im in scaled:
        # vertically align top (you may change to center)
        composed.paste(im, (x, 0), im if im.mode == "RGBA" else None)
        x += im.width + gap

    return composed.convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Create animation of LaTeX document across git history commits.")
    parser.add_argument("repo", help="Path to the git repository")
    parser.add_argument("--tex", type=str, nargs="+", default=["main.tex"], help="Main .tex file path relative to repo root (default: main.tex)")
    parser.add_argument("--out", default="history_anim.gif", help="Output animation filename (gif recommended)")
    parser.add_argument("--out-dir", default="latex_history_out", help="Directory to store intermediate PNGs and PDFs")
    parser.add_argument("--max-pages", type=int, default=10, help="Max pages to show side-by-side (default 10)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for pdftoppm rendering (default 150)")
    parser.add_argument("--frame-duration", type=float, default=1.0, help="Frame duration (seconds) for GIF (default 1.0)")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary build directories")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    repo_path = Path(args.repo).resolve()
    if not repo_path.exists():
        logging.error("Repo path does not exist: %s", repo_path)
        sys.exit(2)

    ensure_tools_exist()

    files = args.tex
    tex_rel = Path(files[0])
    tex_abs = repo_path / tex_rel
    if not tex_abs.exists():
        logging.error("Specified tex file not found in repo: %s", tex_abs)
        sys.exit(2)

    # remember current branch/commit to restore later
    logging.info("Recording current git HEAD to restore later.")
    try:
        res = run(["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True)
        original_branch = res.stdout.strip()
    except Exception:
        # detached? get hash
        res = run(["git", "-C", str(repo_path), "rev-parse", "HEAD"], capture_output=True)
        original_branch = res.stdout.strip()
    logging.info("Original HEAD: %s", original_branch)

    commits = get_commits_touching_file(repo_path, files)
    if not commits:
        logging.error("No commits found touching the file(s) %s", files)
        sys.exit(2)

    outdir = Path(args.out_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    composed_pngs = []

    for idx, commit in enumerate(commits, start=1):
        short = commit[:8]
        logging.info("[%d/%d] Processing commit %s", idx, len(commits), short)
        try:
            # checkout commit (detached)
            run(["git", "-C", str(repo_path), "checkout", "--quiet", commit])

            # We'll build in a temporary working directory inside the repo to avoid messing up repo files
            with tempfile.TemporaryDirectory(prefix="latex_build_", dir=str(repo_path)) as workdir:
                # copy the tex file and any local includes? Simpler: run build in repo root, but specify output dir inside temp.
                # Some builds rely on relative file paths; running in repo root is safest.
                build_outdir = Path(workdir) / "out"
                build_outdir.mkdir(parents=True, exist_ok=True)
                try:
                    pdf_path = build_latex(repo_path, tex_rel, repo_path, build_outdir)
                except Exception as e:
                    logging.warning("Build failed for commit %s : %s", short, e)
                    # skip this commit but continue
                    continue

                # convert to png pages using pdftoppm
                png_prefix = Path(workdir) / "page"
                try:
                    pages = pdf_to_png_pages(pdf_path, png_prefix, dpi=args.dpi, max_pages=args.max_pages)
                except Exception as e:
                    logging.warning("pdftoppm failed for commit %s: %s", short, e)
                    continue

                if not pages:
                    logging.warning("No pages produced for commit %s", short)
                    continue

                # compose side-by-side
                try:
                    composed_img = compose_side_by_side(pages, max_pages=args.max_pages, max_height=1200, gap=8)
                except Exception as e:
                    logging.warning("Failed to compose PNG for commit %s: %s", short, e)
                    continue

                out_png = outdir / f"composed_{idx:04d}_{short}.png"
                composed_img.save(out_png, format="PNG")
                logging.info("Wrote %s", out_png)
                composed_pngs.append(out_png)

            # optionally, small safety sleep? Not necessary.

        finally:
            # after processing each commit, do not restore original branch yet — do that at the end.
            pass

    # restore original HEAD
    try:
        run(["git", "-C", str(repo_path), "checkout", "--quiet", original_branch])
        logging.info("Restored original HEAD: %s", original_branch)
    except Exception as e:
        logging.warning("Could not restore original HEAD (%s): %s. You may need to checkout manually.", original_branch, e)

    if not composed_pngs:
        logging.error("No composed PNGs were generated. Exiting.")
        sys.exit(2)

    # Build GIF animation
    logging.info("Building animation %s", args.out)
    frames = []
    for p in composed_pngs:
        img = imageio.imread(str(p))
        frames.append(img)

    # unify width of all frames
    maxwidth = max([f.shape[1] for f in frames])
    for _ in range(len(frames)):
        f = frames.pop(0)
        if f.shape[1] < maxwidth:
            sh = np.zeros((f.shape[0], maxwidth - f.shape[1], f.shape[2]), np.uint8)
            f = np.concatenate((f, sh), axis=1)
        frames.append(f)

    # save gif
    output_anim = Path(args.out)
    try:
        imageio.mimsave(str(output_anim), frames, duration=args.frame_duration)
        logging.info("Animation saved to %s", output_anim)
    except Exception as e:
        logging.error("Failed to write animation: %s", e)
        sys.exit(1)

    logging.info("Done. Intermediate files are in %s", outdir)
    if args.keep_temp:
        logging.info("Note: script was asked to keep temporary build dirs, but this script uses ephemeral temps inside the repo; see %s for composed images.", outdir)


if __name__ == "__main__":
    main()
