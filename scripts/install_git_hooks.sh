#!/bin/bash
# ICR Git Hooks Installer
# Installs post-checkout and post-merge hooks to auto-reindex after git operations
#
# Usage: ./scripts/install_git_hooks.sh [--uninstall]

set -e

HOOKS_DIR=".git/hooks"
ICR_HOOK_MARKER="# ICR-AUTO-REINDEX"

# Check if we're in a git repo
if [ ! -d ".git" ]; then
    echo "Error: Not in a git repository root"
    exit 1
fi

# Check if ICR is installed
if [ ! -d ".icr" ]; then
    echo "Error: ICR not installed. Run install.sh first."
    exit 1
fi

install_hook() {
    local hook_name=$1
    local hook_path="$HOOKS_DIR/$hook_name"

    # Create hooks directory if needed
    mkdir -p "$HOOKS_DIR"

    # Check if hook already has ICR marker
    if [ -f "$hook_path" ] && grep -q "$ICR_HOOK_MARKER" "$hook_path"; then
        echo "  $hook_name: Already installed"
        return
    fi

    # Create or append to hook
    if [ -f "$hook_path" ]; then
        # Append to existing hook
        echo "" >> "$hook_path"
        echo "$ICR_HOOK_MARKER" >> "$hook_path"
        cat >> "$hook_path" << 'HOOKEOF'
# Trigger ICR incremental reindex after git operation
if [ -d ".icr" ] && [ -f ".icr/venv/bin/icd" ]; then
    echo "ICR: Checking for changed files..."
    .icr/venv/bin/python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, str(Path('.icr/venv/lib/python3.11/site-packages')))
sys.path.insert(0, str(Path('.icr/venv/lib/python3.10/site-packages')))
sys.path.insert(0, str(Path('.icr/venv/lib/python3.12/site-packages')))
try:
    from icd.indexing.incremental import check_staleness, run_incremental_reindex
    report = check_staleness(Path('.'), max_stale_files=50)
    if report.needs_reindex:
        print(f'ICR: Reindexing {report.stale_files + report.new_files} changed files...')
        stats = run_incremental_reindex(Path('.'), max_files=50)
        print(f'ICR: Reindexed {stats.get(\"files\", 0)} files')
    else:
        print('ICR: Index is up to date')
except Exception as e:
    print(f'ICR: Auto-reindex skipped ({e})')
" 2>/dev/null || echo "ICR: Auto-reindex skipped"
fi
HOOKEOF
        echo "  $hook_name: Added to existing hook"
    else
        # Create new hook
        cat > "$hook_path" << 'HOOKEOF'
#!/bin/bash
# Git hook with ICR auto-reindex

HOOKEOF
        echo "$ICR_HOOK_MARKER" >> "$hook_path"
        cat >> "$hook_path" << 'HOOKEOF'
# Trigger ICR incremental reindex after git operation
if [ -d ".icr" ] && [ -f ".icr/venv/bin/icd" ]; then
    echo "ICR: Checking for changed files..."
    .icr/venv/bin/python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, str(Path('.icr/venv/lib/python3.11/site-packages')))
sys.path.insert(0, str(Path('.icr/venv/lib/python3.10/site-packages')))
sys.path.insert(0, str(Path('.icr/venv/lib/python3.12/site-packages')))
try:
    from icd.indexing.incremental import check_staleness, run_incremental_reindex
    report = check_staleness(Path('.'), max_stale_files=50)
    if report.needs_reindex:
        print(f'ICR: Reindexing {report.stale_files + report.new_files} changed files...')
        stats = run_incremental_reindex(Path('.'), max_files=50)
        print(f'ICR: Reindexed {stats.get(\"files\", 0)} files')
    else:
        print('ICR: Index is up to date')
except Exception as e:
    print(f'ICR: Auto-reindex skipped ({e})')
" 2>/dev/null || echo "ICR: Auto-reindex skipped"
fi
HOOKEOF
        chmod +x "$hook_path"
        echo "  $hook_name: Created"
    fi
}

uninstall_hook() {
    local hook_name=$1
    local hook_path="$HOOKS_DIR/$hook_name"

    if [ ! -f "$hook_path" ]; then
        echo "  $hook_name: Not installed"
        return
    fi

    if ! grep -q "$ICR_HOOK_MARKER" "$hook_path"; then
        echo "  $hook_name: No ICR hook found"
        return
    fi

    # Remove ICR section from hook
    # Create temp file without ICR section
    local temp_file=$(mktemp)
    awk "/$ICR_HOOK_MARKER/{found=1} !found{print} /^fi$/ && found{found=0; next}" "$hook_path" > "$temp_file"

    # Check if anything remains
    if [ $(wc -l < "$temp_file") -le 2 ]; then
        # Only shebang remains, remove the hook
        rm "$hook_path"
        echo "  $hook_name: Removed (was ICR-only)"
    else
        mv "$temp_file" "$hook_path"
        chmod +x "$hook_path"
        echo "  $hook_name: ICR section removed"
    fi
    rm -f "$temp_file"
}

if [ "$1" = "--uninstall" ]; then
    echo "Uninstalling ICR git hooks..."
    uninstall_hook "post-checkout"
    uninstall_hook "post-merge"
    uninstall_hook "post-rewrite"
    echo "Done."
else
    echo "Installing ICR git hooks..."
    echo "These hooks will auto-reindex after git checkout/merge/rebase"
    echo ""
    install_hook "post-checkout"
    install_hook "post-merge"
    install_hook "post-rewrite"
    echo ""
    echo "Git hooks installed. ICR will auto-reindex after:"
    echo "  - git checkout"
    echo "  - git merge"
    echo "  - git pull"
    echo "  - git rebase"
    echo ""
    echo "To uninstall: ./scripts/install_git_hooks.sh --uninstall"
fi
