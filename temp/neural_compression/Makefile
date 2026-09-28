PYTHON ?= python3
SPH_DIR ?= ../sph

.PHONY: check compile smoke smoke-density-only smoke-density-velocity

check: compile
	$(PYTHON) -m unittest discover -v

compile:
	$(PYTHON) -m compileall -q .

smoke: smoke-density-only smoke-density-velocity

smoke-density-only:
	$(MAKE) -C $(SPH_DIR) draw2-density-only
	$(PYTHON) smoke_training.py --variant density --device auto

smoke-density-velocity:
	$(MAKE) -C $(SPH_DIR) draw2-density-velocity
	$(PYTHON) smoke_training.py --variant density_velocity --device auto
