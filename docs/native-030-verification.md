# Native 0.3.0 verification

The October 21 native gate at `0a4f8b6` passes 551 all-target Rust tests,
three doctests, Clippy with warnings denied, and formatting checks. Its frozen
worktree remains clean, all 306 recorded Rust/Cargo/toolchain input hashes are
unchanged, and the four log hashes match the receipt.

The [native attestation](verification-native-030-2026-10-21.json) compares the
complete recorded path sets and LF-normalized SHA-256 values from committed
Git snapshots at `0a4f8b6`, the release-native build `26b2ed5`, the final wheel
build `258ff6b`, and `951642a`. All 306 paths match. Foreign edits in the root
working checkout are excluded from these comparisons.

Both extracted Windows extensions and both wheels' native members have SHA-256
`8fa0323b0b1cf80ab4e35dee8216de497586c3213b8489a79abf5bf492ee3675`.
Loading the final extracted extension reports version `0.3.0`. The final
`258ff6b` wheel has SHA-256
`f378b6aa5b0603054058102a93780f720a0a92f92dc66bbf7f029252883bdd5c`.
The committed reference pins and the unchanged reference checkout remain at
`b0dd01be61d517c387e234903a6e2cc38c057ea2`.

This is a native source-quality and artifact-provenance attestation. Cargo ran
debug-profile test executables; its tests did not execute the production
extension inside the wheel. Python/GPU numerical checks, package publication,
and complete main-package behavioral coverage require their own evidence.
