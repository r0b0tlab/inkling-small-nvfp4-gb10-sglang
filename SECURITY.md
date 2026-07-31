# Security contract

This package is intentionally weight-free and non-live.

* No tracked artifact may contain credentials, private addresses, private
  paths, hostnames, users, raw logs, or model payloads.
* Model roots are exact offline snapshots and must be mounted read-only.
* Runtime and image identities are exact SHA/revision contracts.
* Renderers never execute. Launch requires explicit admission and a separate
  execute opt-in. Stop accepts only bounded, owned container names.
* Docker commands use argument arrays, host networking, a read-only bind mount,
  no shell interpolation, no broad cleanup, and no environment secrets.
* Evidence validators reject `PASS` in the non-live foundation and preserve
  `NO_VERDICT` until real live evidence is independently bound.
* The public scan is fail-closed for tracked symlinks, oversized files,
  invalid UTF-8 text, NUL bytes, secret-shaped assignments, private IPs, and
  user-home paths.

The scanner itself uses descriptor-rooted no-follow reads for snapshot
verification and checks file identity before and after hashing. It does not
claim to defeat a malicious same-UID process after verification returns.
