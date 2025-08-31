# NTU-KTP Data Quality Project - Makefile

.PHONY: help clean

# Colors for terminal output
GREEN := \033[32m
BLUE := \033[34m
RESET := \033[0m

# Default target
help: ## Show available commands
	@echo "$(BLUE)NTU-KTP Data Quality Project$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Cleanup
clean: ## Clean build artifacts and cache
	@echo "$(GREEN)Cleaning up...$(RESET)"
	rm -rf build/ dist/ */*.egg-info/
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ htmlcov/ .coverage
