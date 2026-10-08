.PHONY: sync setup run teleop-validate stop shutdown voltage imu sim viewer gamepad-headless-enable gamepad-headless-disable

HOST ?= microban
ID ?=

sync:
	rsync -avz \
		--exclude='.git' \
		--exclude='.venv' \
		--exclude='.pytest_cache' \
		--exclude='__pycache__' \
		--exclude='cad' \
		--exclude='docs' \
		--exclude='logs' \
		--exclude='src/debug' \
		--exclude='src/sim' \
		--exclude='src/model/mjcf' \
		./ $(HOST):microban

setup: sync
	ssh $(HOST) "bash -l -c 'cd microban && uv sync --frozen'"

sim:
	PYTHONPATH=src uv run --group sim src/sim/sim_main.py --hz 50

viewer:
	PYTHONPATH=src uv run src/sim/viewer_main.py --hz 25

run: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/main.py'"

# Validate the installed walk and get-up policies (contract, manifest and
# startup self-test) with the exact local runtime, sync the same checkout and
# lockfile, materialize the Pi environment, then repeat the same CPU-only
# validation on the Pi (docs/policies.md). This never opens the motor bus.
teleop-validate:
	PYTHONPATH=src uv run --locked python tools/validate_policies.py src/agents
	$(MAKE) setup HOST=$(HOST)
	ssh $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python tools/validate_policies.py src/agents'"

stop:
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/stop.py'"

voltage: sync
	ssh $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/voltage.py $(ID)'"

imu: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/imu.py'"

shutdown:
	ssh -tt $(HOST) "sudo shutdown -h now"

# Opt-in headless mode: a service launches the control loop when START is held 2s on
# the gamepad (no SSH needed); START stops it. See docs/gamepad.md.
gamepad-headless-enable: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && sudo bash systemd/install-gamepad-daemon.sh'"

gamepad-headless-disable:
	ssh -tt $(HOST) "bash -l -c 'cd microban && sudo bash systemd/install-gamepad-daemon.sh --uninstall'"
