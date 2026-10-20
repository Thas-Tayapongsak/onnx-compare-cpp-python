# ONNX compare: Python vs C++

This project compares the behavior of ONNX models in two pipelines: one in Python and one in C++.
Both pipelines run the same model on the same input and write their results to files.
A third program, `compare`, runs both pipelines and checks their results against each other.

## Tools

This project was developed for Windows 11.

All tools must be on your PATH.

| Tool | Version |
| --- | --- |
| uv | 0.12.4 |
| Python | 3.12.10 |
| CMake | 4.4.4 |
| Ninja | 1.13.2 |
| clang | 23.1.3 |
| Visual Studio 2022 Build Tools | 17.14.37710.0 |

## Dependencies

### C++

CMake fetches ONNX Runtime, OpenCV, miniz, and nlohmann/json.

| Dependency | Version |
| --- | --- |
| ONNX Runtime | 1.31.0 |
| OpenCV | 5.0.0 |
| miniz | 3.1.2 |
| nlohmann/json | 3.12.0 |

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

The raw pipeline command is:

```text
uv run --project python/pipeline py-pipeline --model <model folder> --image <image> --output <output folder>
```

`--model` and `--image` are required. `--output` defaults to `runs`.
Both pipelines support `-h` and `--help`.
The pipeline writes `outputs.npz` and `meta.json` directly into the folder it
receives.

The raw comparison command is:

```text
uv run --project python/compare compare --model <model folder> --image <image> --output <run root>
```

The compare program takes the same three flags. `--output` defaults to `runs`
and is the root that holds the run folder. The program creates
`<run root>/<YYYYMMDDTHHMMSSZ>-<model folder name>/`, runs both pipelines into
its `python/` and `cpp/` folders, and writes `report.json`.

`scripts/compare.ps1` passes `--model`, `--image`, and `--output` to `compare`.
Run it from the repository root so relative paths resolve there.

## Models and input images

`onnx/` holds one folder per model, with `model.onnx` and its config files. Other JSON files are training metadata which may be useful.
This is the form the models take in practical use.
Treat the folder as read-only.

`images/` holds the input images for runs. One run takes one image.
Treat the folder as read-only too.

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

### What the result files contain

`outputs.npz` holds the session output: one array per ONNX output name.
It has these keys and shapes:

| Key | Shape | Meaning |
| --- | --- | --- |
| `num_predictions` | `[1]` | prediction count `N` |
| `scores` | `[N]` | confidence per prediction |
| `labels` | `[N]` | class id per prediction |
| `boxes` | `[N, 4]` | absolute boxes, scaled to `H' x W'` |
| `masks` | `[N, H', W']` | one mask per prediction |

`N` is the number of predictions. `H'` and `W'` are the output mask height and
width.

`meta.json` holds the run details: `pipeline`, `model`, `input_image`,
`confidence_threshold`, `nms_iou_threshold`, `target_sizes`, `created_utc`,
`setup_ms`, `preprocess_ms`, `inference_ms`, `postprocess_ms`,
`duration_ms`, `execution_provider`, `cpu`, `gpu`, `setup_ram_gb`,
`preprocess_ram_gb`, `inference_ram_gb`, `postprocess_ram_gb`, `ram_gb`,
`setup_vram_gb`, `preprocess_vram_gb`, `inference_vram_gb`,
`postprocess_vram_gb`, and `vram_gb`.
`execution_provider` is the ONNX Runtime execution provider name the run used,
for example `CPUExecutionProvider`.
The `*_ms` fields are integer milliseconds. `setup_ms` is measured once per
run. The preprocess, inference, and postprocess fields are medians across the
inference payloads, one sample per payload. `duration_ms` is the median
payload time from preprocess through postprocess; it excludes setup.

`cpu` is the machine CPU name. `gpu` is the CUDA device used by the process,
or `null`. The RAM fields are the working set of the pipeline process, not the
total machine memory. The memory sampler takes a sample every 50 ms. A stage
RAM field is the highest sample during that stage. The payload-stage values
are maxima across the payloads, and `ram_gb` is the whole-run sampled peak
through output writing. The 50 ms interval limits interference with stage
timings; a shorter allocation can be missed. The VRAM fields are the VRAM used
by the process. They are `0` when no CUDA device is used. The `_gb` fields use
2^30 bytes.

`report.json` holds the comparison result. For now it has one boolean per
`outputs.npz` key. A boolean is `true` when the two arrays are exactly equal.
A `false` value is a valid comparison result; the compare program exits zero
when both pipelines succeed.

Load `outputs.npz` with `numpy.load` and `meta.json` with `json.load`.
