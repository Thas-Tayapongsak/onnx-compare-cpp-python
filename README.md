# ONNX compare: Python vs C++

This project compares the behavior of ONNX models in two pipelines: one in Python and one in C++.
Both pipelines run the same model on the same input and write their results to files.
A third program, `compare`, runs both pipelines and checks their results against each other.

## Tools you need

- Windows. This project is Windows only.
- `uv` for Python packages and environments. Python 3.12.
- CMake, Ninja, clang (LLVM), and Visual Studio 2022 Build Tools for the C++ side.
- git.

All tools must be on your PATH.
The pinned versions are in `_BLUEPRINT.md`.

## Workspace layout

The official workspace tree lives in `_BLUEPRINT.md`. That file is local and is
not tracked in git. The tree lists every root file and every project folder.

The two Python projects have separate environments.
Comparison runs the pipeline as a separate process and reads its result files.

## First-time setup

```bash
uv sync --project python/pipeline
uv sync --project python/compare
scripts/build.ps1
```

## Standard commands

| Command | What it does |
| --- | --- |
| `scripts/build.ps1` | Configures and builds the C++ side with CMake and Ninja |
| `scripts/compare.ps1` | Runs both pipelines on a model and writes the comparison |
| `scripts/lint.ps1` | Checks code style with `ruff` and `clang-format` |

The scripts wrap the raw `cmake` and `uv run` commands.
Use raw commands to diagnose a script failure. Report the difference.

## Models and input images

`onnx/` holds one folder per model, with `model.onnx` and its config files. Other JSON files are training metadata which may be useful.
This is the form the models take in practical use.
Treat the folder as read-only. Nothing in it goes into git.

`images/` holds the input images for runs. One run takes one image.
Treat the folder as read-only too. Nothing in it goes into git.

## Run output

Each run writes to its own folder:

```text
runs/<YYYYMMDDTHHMMSSZ>-<model folder name>/
  python/            outputs.npz, meta.json
  cpp/               outputs.npz, meta.json
  report.json        the comparison result
```

The timestamp is UTC. The format is file-safe on Windows, for example
`20261018T040454Z`.

Both pipelines write the same two file names and the same keys inside them.
This is the rule the comparison program relies on. The exact keys and shapes
are in `_BLUEPRINT.md`.
Old runs are kept. Nothing is deleted automatically.

### What the result files contain

`outputs.npz` holds the session output: one array per ONNX output name,
with the prediction count, scores, labels, boxes, and masks.

`meta.json` holds the run details: pipeline name, model, input image,
thresholds, target size, run time, and the machine's CPU, GPU, and memory.
