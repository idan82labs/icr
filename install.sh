#!/bin/bash
# ICR Installer - One command to give Claude perfect memory of your codebase
# Usage: curl -fsSL https://raw.githubusercontent.com/idan82labs/icr/main/install.sh | bash

set -e

echo "Installing ICR..."

# Check Python version
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
    echo "Error: Python 3.10+ is required"
    echo "Install with: brew install python@3.11"
    exit 1
fi

echo "Using $PYTHON"

# Create ICR directory
mkdir -p .icr

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON -m venv .icr/venv

# Install packages
echo "Installing ICR packages (this may take a minute)..."
.icr/venv/bin/pip install --quiet --upgrade pip
.icr/venv/bin/pip install --quiet "icd @ git+https://github.com/idan82labs/icr.git#subdirectory=icd"
.icr/venv/bin/pip install --quiet "ic-mcp @ git+https://github.com/idan82labs/icr.git#subdirectory=ic-mcp"

# Create MCP config
echo "Configuring MCP server..."
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
echo "Created .icrignore"
fi

# Index the codebase
echo "Indexing your codebase..."
.icr/venv/bin/icd index --repo-root .

# Add to gitignore
if [ -f .gitignore ]; then
    if ! grep -q "^\.icr/$" .gitignore; then
        echo ".icr/" >> .gitignore
    fi
else
    echo ".icr/" > .gitignore
fi

echo ""
echo "====================================="
echo "  ICR installed successfully!"
echo "====================================="
echo ""
echo "Restart Claude Code, then just ask questions:"
echo "  - 'How does the auth system work?'"
echo "  - 'Where is the database connection?'"
echo "  - 'Find all usages of UserService'"
echo ""
echo "Claude now has perfect memory of your codebase."
