# HELP COMMANDS
help: ## show this help
	@echo 'usage: make [target] [option]'
	@echo ''
	@echo 'Common sequence of commands:'
	# @echo '- make test'
	@echo '- make lint'


# .PHONY : test
# test: ## runs the application tests
	# @ $(RUN) py.test -n 2

.PHONY: lint
lint: ## runs linters in updated code
	@ make black
	@ make isort
	@ make flake8

.PHONY: black
black: ## runs black in updated code
	@ /bin/sh -c git diff --diff-filter=ACM dev --name-only | grep '.*.py' | xargs -r black"

.PHONY: flake8
black: ## runs flake8 in updated code
	@ /bin/sh -c git diff --diff-filter=ACM dev --name-only | grep '.*.py' | xargs -r flake8"

.PHONY: isort
isort: ## runs isort over the updated code
	@ /bin/sh -c git diff --diff-filter=ACM dev --name-only | grep '.*.py' | xargs -r isort"