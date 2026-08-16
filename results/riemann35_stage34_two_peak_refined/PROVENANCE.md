# Stage 34 publication provenance

The accepted refined run reports `git_head=809553e744d76890cdf7e4482d030c744dd7e28a`.
That hash is the clean dependency baseline from which the isolated Stage-34
runner was executed; the runner was not yet tracked at generation time.

The recoverable publication commit containing both the exact runner and the
archived result artifacts is:

`9929117dc2f827f13c5815ab0f887f0ea894798a`

The runner is
`riemann35_patch/stage34_two_peak/run_two_peak_audit.py`, with SHA-256:

`295eba165900da8a35e27e0f5dd77d8c6d0d225c02993d43979dd2facb91ddac`

This matches the `source_sha256` stored in the refined JSON and NPZ. The two
CSV files were normalized from CRLF to LF before publication; their headers,
rows, ordering, and numeric payloads are unchanged.
