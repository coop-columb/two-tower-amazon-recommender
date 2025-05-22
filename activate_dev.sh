#!/bin/bash
# Two-Tower Recommender Development Environment Activation
# Streamlined approach leveraging proven working components

set -e  # Exit on any error

# Validate project directory context
if [[ ! -f ".python-version" ]] || [[ ! -d "venv" ]]; then
    echo "❌ Not in Two-Tower Recommender project root directory"
    echo "   Expected files: .python-version, venv/"
    return 1
fi

PROJECT_ROOT="$(pwd)"
VENV_ACTIVATE="$PROJECT_ROOT/venv/bin/activate"

# Validate virtual environment integrity
if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "❌ Virtual environment activation script not found"
    echo "   Expected: $VENV_ACTIVATE"
    echo "   Run: python -m venv venv"
    return 1
fi

# Execute standard virtual environment activation
echo "🔧 Activating virtual environment..."
source "$VENV_ACTIVATE"

# Verify activation success before proceeding
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "❌ Virtual environment activation failed"
    return 1
fi

# Apply PATH precedence correction (proven working method)
echo "🔧 Configuring tool PATH precedence..."
export PATH="$PROJECT_ROOT/venv/bin:$PATH"

# Set project-specific environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export MLFLOW_TRACKING_URI="file://$PROJECT_ROOT/mlruns"
export WANDB_PROJECT="two-tower-recommender"

# Comprehensive environment validation
echo "✅ Development environment activated successfully:"
echo "   Project Root: $PROJECT_ROOT"
echo "   Virtual Environment: $VIRTUAL_ENV"
echo "   Python: $(which python) [$(python --version)]"
echo "   Pyenv Local: $(cat .python-version)"

# Development tools validation with status indicators
echo "   Development Tools:"
declare -a dev_tools=("black" "isort" "flake8" "mypy" "pytest" "pre-commit")

for tool in "${dev_tools[@]}"; do
    if command -v "$tool" >/dev/null 2>&1; then
        tool_path=$(which "$tool")
        if [[ "$tool_path" == *"venv/bin"* ]]; then
            echo "     ✅ $tool"
        else
            echo "     ⚠️  $tool: $tool_path"
        fi
    else
        echo "     ❌ $tool: not installed"
    fi
done

echo ""
echo "🚀 Ready for Two-Tower Recommender development!"
echo "   Use 'deactivate' to exit environment"
echo "   Use 'python validate_environment.py' for comprehensive validation"
