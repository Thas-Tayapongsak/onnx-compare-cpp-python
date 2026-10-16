# ONNX compare: Python vs C++

This project compares the behavior of ONNX models in two pipelines: one in Python and one in C++.
Both pipelines run the same model on the same input and write their results to files.
A third program, `compare`, runs both pipelines and checks their results against each other.

**Status:** work in progress. The workspace layout exists. The pipelines are still stubs.

## Tools you need

- Windows. This project is Windows only.
- `uv` for Python packages and environments. Python 3.12.
- CMake, Ninja, clang (LLVM), and Visual Studio 2022 Build Tools for the C++ side.
- git.

All tools must be on your PATH.

## Workspace layout

```text
onnx/                      model folders (read-only data, not in git)
images/                    input images for runs (read-only data, not in git)
python/
  pipeline/                uv project "py-pipeline" (the Python pipeline)
  compare/                 uv project "compare" (the comparison program)
cpp/
  CMakeLists.txt           one CMake project
  CMakePresets.json        the standard configure and build command
  src/                     library "pipeline_core" + executable "cpp_pipeline"
  build/                   build output (not in git)
scripts/
  build.ps1                builds the C++ side
  compare.ps1              runs the comparison program
  lint.ps1                 checks code style
runs/                      one folder per run (not in git)
README.md
.gitignore  .gitattributes  .editorconfig  .clang-format  .clangd  ruff.toml
```

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
You can run the raw commands yourself if you prefer.

## Models and input images

`onnx/` holds one folder per model, with `model.onnx` and its config files. Other JSON files are training metadata which may be useful.
This is the form the models take in practical use.
Treat the folder as read-only. Nothing in it goes into git.

`images/` holds the input images for runs. One run takes one image.
Treat the folder as read-only too. Nothing in it goes into git.

## Run output

Each run writes to its own folder:

```text
runs/<UTC timestamp>-<model name>/
  python/            outputs.npz, meta.json
  cpp/               outputs.npz, meta.json
  report.json        the comparison result
```

Both pipelines write the same two file names and the same keys inside them.
This is the rule the comparison program relies on.
Old runs are kept. Nothing is deleted automatically.

### What the result files contain

`outputs.npz` holds the session output: one array per ONNX output name,
with the prediction count, scores, labels, boxes, and masks.

`meta.json` holds the run details: pipeline name, model, input image,
thresholds, target size, run time, and the machine's CPU, GPU, and memory.
