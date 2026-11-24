# Lessons learned

Memo for the v4.28.0 project-structure work. Date: 24 November 2026.

## What shipped

The public command is still `python video_orientation_detector.py`. The implementation is now four modules: the facade, `svod_detector.py`, `svod_bootstrap.py`, and `svod_state.py`. GitHub keeps only `main`. Pytest covers 216 tests. On 24 November 2026, `testing/standard_batch_test.py --time-limit 10` scored 22/22 and 15/15 on orientation and direction.

## Lessons

1. **Delete a branch only after checking what it uniquely contains.** `feature/advanced-detection-v4.25`, `feature/performance-optimization-v4.25`, and `yolov10-testing` were already in `main`. `feature/boril` was not. Its only extra commit added three September 2025 YOLOv4 scripts. Those scripts conflict with the rule that manual video tests use the three scripts in `testing/`, so the branch was deleted rather than merged.

2. **A push that deletes several branches fails entirely when any named ref is already gone.** Delete the refs that still exist, then `git fetch --prune`.

3. **Moving the detector is safe only when method lookup stays the same.** `YOLO_AVAILABLE` and the other flags are assigned from the CLI as attributes of `video_orientation_detector`. The class reads them from `svod_state`, and the facade forwards those attribute reads and writes. A bool copied into the new module at import time would go stale.

4. **Do not reflow `svod_detector.py` with a homemade wrapper.** One wrap shortened `box.conf` to `box.con`, so a mocked confidence of 0.85 was no longer read, and another wrap turned an f-string into a call. Black plus a method-by-method AST compare against the previous file is the check. The restored methods differ from v4.27.1 only in those flag lookups, the removal of an unused `global`, and one redundant `import math`.

5. **A workflow condition on `github.event.inputs` does not run for a normal push.** The old "full video" CI step required a manual dispatch input, so pushes never executed it. CI now runs pytest. The 37 reference videos stay on this machine.

6. **Match the CI matrix to wheels that exist.** PyTorch 2.8 and OpenVINO 2024.6 publish 3.11 and 3.12 wheels for Linux, Windows, and Apple Silicon. They do not publish a macOS x86_64 torch wheel, so Intel macOS only runs `--version`.

7. **Coverage has to follow the code.** After the split, measuring only `video_orientation_detector.py` would have scored the 93-line facade and missed the detector. The gate measures all seven SVOD modules and was 32.86% on 24 November 2026.

8. **Keep the lint baseline honest.** The old per-file ignore list hid unused imports and bare `except` clauses. Those are fixed, and `tests/` is linted. `E501` remains ignored because many debug strings are longer than 120 characters and Black does not break them.

9. **Use the repository author address.** `pyproject.toml` had `boris@example.com`. The address on recent commits is `boris.jalov@gmail.com`.

10. **Say which reference run was actually executed.** The 24 November 2026 check used `--time-limit 10` on all 37 videos. The speed and memory numbers in `performance_baselines/` are still the v4.27.1 measurement, because this release did not change detection math.
