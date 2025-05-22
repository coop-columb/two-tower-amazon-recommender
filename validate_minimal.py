#!/usr/bin/env python3
"""
Minimal environment validation with crash prevention mechanisms.
Conservative implementation avoiding subprocess complexity.
"""

import os
import sys
from pathlib import Path

def safe_print(message: str, status: str = "INFO") -> None:
    """Thread-safe printing with status indicators."""
    symbols = {"SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}
    symbol = symbols.get(status, "ℹ️")
    print(f"{symbol} {message}")

def validate_basic_environment() -> bool:
    """Conservative environment validation without subprocess calls."""
    safe_print("Two-Tower Recommender Environment Validation - Minimal", "INFO")
    print("-" * 50)
    
    issues = []
    
    # Check current directory context
    current_dir = Path.cwd()
    if current_dir.name == "two-tower-amazon-recommender":
        safe_print(f"Project directory: {current_dir}", "SUCCESS")
    else:
        safe_print(f"Unexpected directory: {current_dir}", "ERROR")
        issues.append("Not in project root directory")
    
    # Check virtual environment activation
    venv_path = os.environ.get('VIRTUAL_ENV', '')
    if venv_path and 'two-tower-amazon-recommender' in venv_path:
        safe_print(f"Virtual environment: {venv_path}", "SUCCESS")
    else:
        safe_print("Virtual environment not properly activated", "ERROR")
        issues.append("Virtual environment activation required")
    
    # Check essential project files
    essential_files = [
        ".python-version",
        "activate_dev.sh",
        "venv/bin/activate",
        ".vscode/settings.json"
    ]
    
    for file_path in essential_files:
        full_path = current_dir / file_path
        if full_path.exists():
            safe_print(f"File exists: {file_path}", "SUCCESS")
        else:
            safe_print(f"Missing file: {file_path}", "ERROR")
            issues.append(f"Missing: {file_path}")
    
    # Check Python module availability without imports
    python_path = sys.executable
    if 'venv' in python_path:
        safe_print(f"Python executable: {python_path}", "SUCCESS")
    else:
        safe_print(f"Python not from venv: {python_path}", "WARNING")
    
    # Check PATH configuration
    path_env = os.environ.get('PATH', '')
    venv_bin_in_path = any('venv/bin' in p for p in path_env.split(':'))
    if venv_bin_in_path:
        safe_print("PATH includes venv/bin", "SUCCESS")
    else:
        safe_print("PATH missing venv/bin precedence", "WARNING")
    
    # Summary report
    print("\n" + "=" * 50)
    if not issues:
        safe_print("Minimal validation completed successfully!", "SUCCESS")
        safe_print("Environment ready for development", "SUCCESS")
        return True
    else:
        safe_print(f"Found {len(issues)} issues:", "ERROR")
        for issue in issues:
            print(f"   - {issue}")
        return False

def main():
    """Main execution with exception handling."""
    try:
        success = validate_basic_environment()
        sys.exit(0 if success else 1)
    except Exception as e:
        safe_print(f"Validation error: {str(e)}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()
