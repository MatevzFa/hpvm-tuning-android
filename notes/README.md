## Notes

For succesfull tuning via Singularity on SLING, the following files have to be patched. 

The **Target file** path assumes you have a sandboxed HPVM Singularity image in directory `hpvm.sb`. You should correct the paths to your use case.

| File                 | Target file                                                           |
| -------------------- | --------------------------------------------------------------------- |
| `onnx_simplifier.py` | `hpvm.sb/venv/lib/python3.6/site-packages/onnxsim/onnx_simplifier.py` |
| `approxapp.py`       | `hpvm.sb/venv/lib/python3.6/site-packages/predtuner/approxapp.py`     |
