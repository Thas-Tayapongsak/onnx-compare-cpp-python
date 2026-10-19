# compare

Runs the Python and C++ ONNX pipelines on one model and image, then compares
their `outputs.npz` files.

Run it from the repository root:

```text
uv run --project python/compare compare --model <model folder> --image <image> --output <run root>
```

`--output` defaults to `runs`. The program creates
`<run root>/<YYYYMMDDTHHMMSSZ>-<model folder name>/`, writes the pipeline
results into its `python/` and `cpp/` folders, and writes `report.json` in the
run folder. The report has one boolean per `outputs.npz` key, from exact array
equality.
