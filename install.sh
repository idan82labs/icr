#!/bin/bash
# ICR Installer - One command to give Claude perfect memory of your codebase
# Usage: curl -fsSL https://raw.githubusercontent.com/idan82labs/icr/main/install.sh | bash

set -e

echo "========================================"
echo "  ICR Installer"
echo "========================================"
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
PYTHON=""
for py in python3.12 python3.11 python3.10 python3; do
    if command -v $py &> /dev/null; then
        version=$($py -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON=$py
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "  Error: Python 3.10+ is required"
    echo "  Install with: brew install python@3.11"
    exit 1
fi

echo "  Found $PYTHON ($version)"

# Create ICR directory
echo ""
echo "[2/6] Creating ICR directory..."
mkdir -p .icr
echo "  Created .icr/"

# Create virtual environment
echo ""
echo "[3/6] Creating virtual environment..."
$PYTHON -m venv .icr/venv
echo "  Created .icr/venv/"

# Install packages
echo ""
echo "[4/6] Installing ICR packages..."
echo "  This may take 1-2 minutes on first install..."
.icr/venv/bin/pip install --quiet --upgrade pip 2>&1 | grep -E "(ERROR|error)" || true

echo "  Installing icd (indexer)..."
if .icr/venv/bin/pip install --quiet "icd @ git+https://github.com/idan82labs/icr.git#subdirectory=icd" 2>&1 | grep -E "(ERROR|error)"; then
    echo "  Warning: icd installation had errors"
else
    echo "  icd installed successfully"
fi

echo "  Installing ic-mcp (MCP server)..."
if .icr/venv/bin/pip install --quiet "ic-mcp @ git+https://github.com/idan82labs/icr.git#subdirectory=ic-mcp" 2>&1 | grep -E "(ERROR|error)"; then
    echo "  Warning: ic-mcp installation had errors"
else
    echo "  ic-mcp installed successfully"
fi

# Create MCP config
echo ""
echo "[5/6] Configuring MCP server..."
cat > .mcp.json << 'MCPEOF'
{
  "mcpServers": {
    "icr": {
      "type": "stdio",
      "command": ".icr/venv/bin/ic-mcp",
      "args": ["--repo-root", ".", "--log-file", ".icr/mcp.log", "--log-level", "WARNING"]
    }
  }
}
MCPEOF
echo "  Created .mcp.json"

# Create default config
cat > .icr/config.yaml << 'CFGEOF'
embedding:
  model_name: all-MiniLM-L6-v2
  dimension: 384

retrieval:
  weight_embedding: 0.4
  weight_bm25: 0.3
  weight_recency: 0.1
  weight_contract: 0.1

pack:
  default_budget_tokens: 8000

watcher:
  enabled: true
  respect_gitignore: true
  respect_icrignore: true
CFGEOF
echo "  Created .icr/config.yaml"

# Create default .icrignore if it doesn't exist
if [ ! -f .icrignore ]; then
cat > .icrignore << 'IGNEOF'
# ICR Ignore File - patterns here are excluded from indexing
# Uses gitignore syntax

# Secrets and credentials
.env
.env.*
*.pem
*.key
credentials.json
secrets.yaml

# Large generated files
*.min.js
*.min.css
*.bundle.js

# Lock files
package-lock.json
yarn.lock
pnpm-lock.yaml
poetry.lock

# Test snapshots
__snapshots__/
*.snap
IGNEOF
echo "  Created .icrignore"
fi

# Index the codebase
echo ""
echo "[6/6] Indexing your codebase..."
echo "  This may take a few minutes for large codebases..."
if .icr/venv/bin/icd index --repo-root . 2>&1 | tail -5; then
    echo "  Indexing complete"
else
    echo "  Warning: Indexing may have encountered issues"
fi

# Validate index creation
echo ""
echo "Validating installation..."
if [ -f ".icd/index.db" ] || [ -f ".icr/index.db" ]; then
    echo "  Index database created successfully"

    # Get index stats if possible
    if command -v sqlite3 &> /dev/null; then
        DB_PATH=""
        if [ -f ".icd/index.db" ]; then
            DB_PATH=".icd/index.db"
        elif [ -f ".icr/index.db" ]; then
            DB_PATH=".icr/index.db"
        fi

        if [ -n "$DB_PATH" ]; then
            FILE_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(DISTINCT file_path) FROM chunks;" 2>/dev/null || echo "?")
            CHUNK_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM chunks;" 2>/dev/null || echo "?")
            echo "  Indexed: $FILE_COUNT files, $CHUNK_COUNT chunks"
        fi
    fi
else
    echo "  Warning: Index database not found"
    echo "  Try running: .icr/venv/bin/icd index --repo-root ."
fi

# Add to gitignore
if [ -f .gitignore ]; then
    if ! grep -q "^\.icr/$" .gitignore; then
        echo ".icr/" >> .gitignore
        echo "  Added .icr/ to .gitignore"
    fi
else
    echo ".icr/" > .gitignore
    echo "  Created .gitignore with .icr/"
fi

# Install git hooks for auto-reindex (if in a git repo)
if [ -d ".git" ]; then
    echo ""
    echo "Optional: Install git hooks for auto-reindex?"
    echo "  This will reindex after git checkout/merge/pull."

    # Check if running interactively
    if [ -t 0 ]; then
        read -p "  Install git hooks? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Download and run git hooks installer
            if [ -f "scripts/install_git_hooks.sh" ]; then
                bash scripts/install_git_hooks.sh
            else
                echo "  Downloading git hooks installer..."
                curl -fsSL https://raw.githubusercontent.com/idan82labs/icr/main/scripts/install_git_hooks.sh | bash
            fi
        else
            echo "  Skipped. Run later with: curl -fsSL https://raw.githubusercontent.com/idan82labs/icr/main/scripts/install_git_hooks.sh | bash"
        fi
    else
        echo "  (Non-interactive mode - skipping)"
        echo "  Run later: curl -fsSL https://raw.githubusercontent.com/idan82labs/icr/main/scripts/install_git_hooks.sh | bash"
    fi
fi

echo ""
echo "========================================"
echo "  ICR installed successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Restart Claude Code to activate ICR"
echo "  2. Ask questions like:"
echo "     - 'How does the auth system work?'"
echo "     - 'Where is the database connection?'"
echo "     - 'Find all usages of UserService'"
echo ""
echo "Auto-reindex:"
echo "  - On session start: ICR checks for changed files"
echo "  - With git hooks: Reindex after checkout/merge/pull"
echo ""
echo "For known symbols, use native Grep/Glob (faster)."
echo "For conceptual questions, ICR shines!"
echo ""
echo "To update ICR: curl -fsSL https://raw.githubusercontent.com/idan82labs/icr/main/install.sh | bash"
echo ""
