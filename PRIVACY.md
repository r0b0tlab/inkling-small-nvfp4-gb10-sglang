# Privacy and public-evidence policy

This repository contains only curated, bounded, claim-bearing evidence. It must not contain:

- credentials, tokens, cookies, authorization headers, or private registry configuration;
- private hostnames, addresses, usernames, home paths, SSH details, or cluster topology values;
- model weights, shard payloads, model caches, generated raw responses, raw BFCL trees, or private logs;
- temporary DNS mappings, mount/export details, process command lines, or raw telemetry streams.

Public aggregates are generated from descriptor-hashed private evidence and are admitted by `scripts/public_safety_scan.py`. Raw evidence remains outside Git.
