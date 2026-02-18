# Security Policy

## Reporting a Vulnerability

Please report suspected vulnerabilities privately to the maintainers.
Avoid opening public issues with exploit details before triage.

Include:

- affected version
- reproduction steps
- impact assessment
- any suggested mitigation

## Supported Versions

The latest minor release on the latest major version is supported for security fixes.

## Supply-Chain Baseline

Use the bundled script to run dependency and SBOM checks:

```bash
./scripts/supply_chain_check.sh
```

This currently includes:

- dependency vulnerability audit via `pip-audit`
- optional SBOM generation via `cyclonedx-py`
