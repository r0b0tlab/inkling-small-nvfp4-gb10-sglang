PYTHON ?= python3

.PHONY: test validate safety

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

validate:
	$(PYTHON) -m compileall -q src scripts tests
	PYTHONPATH=src $(PYTHON) -c "from pathlib import Path; from inkling_release.manifest import load_runtime_manifest; load_runtime_manifest(Path('runtime-manifest.json'))"

safety:
	$(PYTHON) scripts/public_safety_scan.py .
