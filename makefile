VENV_DIR=.venv
VENV_ACTIVATE=$(VENV_DIR)/bin/activate
PYTHON=$(VENV_DIR)/bin/python

.PHONY: help venv lint run clean

help:
	@echo "'make run': Run all examples."
	@echo "'make lint': Run a linter in all code."
	@echo "'make venv': Prepare development environment, use only once."
	@echo "'make clean': Cleans up generated files."
	@echo

venv: $(VENV_ACTIVATE)
$(VENV_ACTIVATE):
	@echo "Creating a virtualenv..."
	@python3 -m venv "$(VENV_DIR)"
	@echo "Installing packages in the virtualenv..."
	@. $(VENV_ACTIVATE); \
		pip3 install --upgrade pip setuptools; \
		pip3 install --upgrade --requirement "requirements.txt"
	@echo "Done!"
	@echo

lint: venv
	@echo "Running linters..."
	@. $(VENV_ACTIVATE); $(PYTHON) -m flake8 *.py
	@. $(VENV_ACTIVATE); $(PYTHON) -m pylint \
		--disable=invalid-name \
		--disable=missing-docstring \
		--disable=too-many-arguments \
		--disable=too-many-locals \
		--disable=duplicate-code *.py
	@echo "Done!"
	@echo

run: venv
	@echo "Running all examples..."
	@. $(VENV_ACTIVATE); $(PYTHON) main.py
	@echo "Done!"
	@echo

clean:
	@echo "Cleaning up generated files..."
	@rm -rf $(VENV_DIR)
	@rm -rf "__pycache__"
	@find . -type f \( -iname "*.py[cod]" \) ! -path "./.git/*" -delete
	@echo "Done!"
	@echo
