#!/bin/bash
# Environment status checker for Two-Tower Recommender

PROJECT_NAME="two-tower-amazon-recommender"

if [[ "$(basename "$(pwd)")" == "$PROJECT_NAME" ]]; then
    if [[ -n "$VIRTUAL_ENV" ]] && [[ "$VIRTUAL_ENV" == *"$PROJECT_NAME"* ]]; then
        echo "✅ Two-Tower Recommender environment active"
        echo "   Virtual Environment: $VIRTUAL_ENV"
        echo "   Python: $(which python)"
    else
        echo "⚠️  In $PROJECT_NAME directory but environment not activated"
        echo "   Run: source activate_dev.sh"
    fi
else
    echo "ℹ️  Not in $PROJECT_NAME project directory"
fi
