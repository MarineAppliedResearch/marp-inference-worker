# Verification — inference watch mode

## Approved scope

The human approved the watch-mode portions of this plan on 2026-09-14. Packaging,
installation, activation, updates, and volunteer compute controls are excluded.

## Automated checks

- **R1, R7-R9** — Run the focused worker watch/channel and runner gate tests.
- **R2, R4-R5, R11-R12** — Run the player's live presenter unit test and production build.
- **R6** — Confirm the server binds loopback and the page payload contains no remote URL or
  credentials through the focused worker test and code inspection.

## System checks

- **R1-R8, R12-R13** — Run one real API job with `params.watch: true` and the worker in
  window mode. Confirm the local window, labels, stable colors, persistence, status, close,
  and headless continuation.
- **R3, R9** — Run the same real model, media, and frame range with watch off and on; compare
  observation artifacts byte-for-byte and record both throughputs.
- Confirm `params.watch` absent and false remain headless.

## Not covered

- Installer size or behavior, clean-machine setup, activation, self-update, volunteer
  controls, remote viewing, and production deployment.

## Results

- **Worker focused tests — PASS.** `33 passed in 18.24s` across the runner gate and ordered
  watch-channel tests.
- **Player presenter/build/host pack — PASS.** The presenter passed `2/2`; all three
  production bundles built; the offline host ZIP contains `live.html` and its standalone
  player bundle. The build retains one pre-existing duplicate `onUnitReady` warning in
  `src/audio-output.js`.
- **Real API/database/Jellyfin/GPU comparison — PASS.** The isolated API resolved the live
  Jellyfin item, a real worker leased both jobs, and PyTorch ran the CAMPA model on the RTX
  4080 SUPER. Watch off processed 300 frames in 8.75 s (34.29 fps); watch on processed the
  same range in 24.20 s (12.39 fps). Both produced the same 20,992-byte observation artifact
  with SHA-256 `78a42ec94592689391e2e92e8cc1db2004a841bc35da88b3714fd10dc4cee1e1`.
- **Display isolation — PASS.** The visible run's substantially lower measured rate and
  matching artifact confirm browser acknowledgements applied backpressure without changing
  scientific output. The loopback and close-to-headless paths are covered by the focused
  channel tests and source review.

The first real submissions exposed three environment/harness problems before the passing
run: the API required the seeder-reported model identity; the local environment had a CPU
PyTorch wheel despite a healthy GPU; and a retried earlier job was ahead of the newly
submitted job. The local CUDA 12.6 wheel was restored and verified with a real CUDA kernel,
and the disposable queue was cleared before the recorded comparison.

An attempted installer build measured 3,058,143,651 bytes. The human rejected that delivery
approach and corrected #11 back to watch mode only. All installer, activation, update, API
migration, and volunteer-control changes were removed from the active branches.
