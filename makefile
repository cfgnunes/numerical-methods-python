VENV_DIR=.venv
VENV_ACTIVATE=$(VENV_DIR)/bin/activate
PYTHON=$(VENV_DIR)/bin/python

.PHONY: help venv lint run clean

help:
	@echo "'make run': Run all examples."
	@echo "'make lint': Run a linter in all code."
	@echo "'make venv': Prepare development environment, use only once."
	@echo "'make clean': Cleans up generated files."

venv:
	@test -d "$(VENV_DIR)" || echo "Creating new virtualenv..."; python3 -m venv "$(VENV_DIR)"
	@echo "Installing packages in the virtualenv..."
	@. "$(VENV_ACTIVATE)"; \
		pip3 install --upgrade pip; \
		pip3 install --upgrade --requirement "requirements.txt"

lint:
	@test -d "$(VENV_DIR)" || make venv
	@flake8 *.py

run:
	@test -d "$(VENV_DIR)" || make venv
	@echo "Running all examples..."
	@$(PYTHON) main.py

clean:
	@rm -rf $(VENV_DIR)
	@rm -rf "__pycache__"
	@find . -type f \( -iname "*.py[cod]" \) ! -path "./.git/*" -delete
