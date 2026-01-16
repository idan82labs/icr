# ICR Troubleshooting Guide

This guide helps diagnose and resolve common issues with ICR installation, operation, and performance.

---

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Installation Issues](#installation-issues)
- [Hook Not Triggering](#hook-not-triggering)
- [Slow Performance](#slow-performance)
- [Index Corruption Recovery](#index-corruption-recovery)
- [Debug Logging](#debug-logging)
- [Common Error Messages](#common-error-messages)
- [Getting Help](#getting-help)

---

## Quick Diagnostics

### Run Health Check

```bash
icr doctor
```

Expected healthy output:
```
ICR Health Check
================

[OK] Configuration found at ~/.icr/config.yaml
[OK] SQLite database accessible
[OK] Vector index initialized
[OK] Embedding backend: local-onnx (all-MiniLM-L6-v2)
[OK] Claude Code hooks configured (user-level)
[OK] MCP server configuration found

Status: HEALTHY
```

### Check Service Status

```bash
# Check if daemon is running
icr status

# View recent logs
icr logs --tail 50

# View metrics
icr metrics --last 1h
```

### Verify Claude Code Integration

```bash
# Check MCP server registration
cat ~/.claude.json | jq '.mcpServers.icr'

# Check hooks configuration
cat ~/.claude/settings.json | jq '.hooks'
```

---

## Installation Issues

### Issue: pip install fails with dependency errors

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement hnswlib>=0.7.0
```

**Solutions:**

1. **Ensure Python 3.10+**:
   ```bash
   python --version
   # Should be 3.10 or higher
   ```

2. **Install build dependencies** (Linux):
   ```bash
   sudo apt-get install python3-dev build-essential
   ```

3. **Install build dependencies** (macOS):
   ```bash
   xcode-select --install
   ```

4. **Try with pip upgrade**:
   ```bash
   pip install --upgrade pip wheel setuptools
   pip install icr
   ```

### Issue: ONNX Runtime installation fails

**Symptoms:**
```
ERROR: onnxruntime requires ...
```

**Solutions:**

1. **Check platform compatibility**:
   ```bash
   # ONNX Runtime has specific platform requirements
   pip install onnxruntime --verbose
   ```

2. **Use CPU-only version** (if GPU issues):
   ```bash
   pip install onnxruntime-cpu
   ```

3. **On Apple Silicon**:
   ```bash
   # Ensure Rosetta is not being used
   arch -arm64 pip install onnxruntime
   ```

### Issue: tree-sitter compilation fails

**Symptoms:**
```
error: command 'gcc' failed
```

**Solutions:**

1. **Install C compiler**:
   ```bash
   # macOS
   xcode-select --install

   # Ubuntu/Debian
   sudo apt-get install build-essential

   # Fedora/RHEL
   sudo dnf install gcc gcc-c++ make
   ```

2. **Install from wheel if available**:
   ```bash
   pip install --only-binary :all: tree-sitter
   ```

### Issue: icr command not found after install

**Solutions:**

1. **Check PATH**:
   ```bash
   echo $PATH
   # Should include ~/.local/bin or similar
   ```

2. **Add to PATH**:
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   export PATH="$HOME/.local/bin:$PATH"
   ```

3. **Use full path**:
   ```bash
   python -m icr doctor
   ```

---

## Hook Not Triggering

### Issue: Context not injected into prompts

**Symptoms:**
- No context pack appears before your messages
- `/ic` commands work but auto-injection does not

**Diagnosis:**

```bash
# Check hooks are configured
cat ~/.claude/settings.json | jq '.hooks'

# Expected output should include UserPromptSubmit
```

**Solutions:**

1. **Re-run configuration**:
   ```bash
   icr configure claude-code --force
   ```

2. **Verify hook script exists**:
   ```bash
   ls -la ~/.icr/scripts/ic-hook-userpromptsubmit.py
   ```

3. **Check script permissions**:
   ```bash
   chmod +x ~/.icr/scripts/ic-hook-*.py
   ```

4. **Test hook manually**:
   ```bash
   echo '{"prompt": "test"}' | python ~/.icr/scripts/ic-hook-userpromptsubmit.py
   ```

### Issue: Hooks configured but nothing happens

**Check Claude Code version:**
```bash
claude --version
# Hooks require Claude Code 1.x or higher
```

**Check hook matcher:**
```json
// In ~/.claude/settings.json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": ".*",  // Should match all prompts
        "hooks": [...]
      }
    ]
  }
}
```

### Issue: Hook errors in logs

**View hook execution logs:**
```bash
# Claude Code logs
tail -f ~/.claude/logs/claude.log | grep -i hook

# ICR logs
icr logs --tail 100 | grep -i hook
```

**Common fixes:**

1. **Python path issues**:
   ```bash
   # Ensure icr is in the Python used by hooks
   which python
   python -c "import icr; print('OK')"
   ```

2. **Working directory issues**:
   ```bash
   # Hook scripts should use absolute paths
   grep -r "os.getcwd" ~/.icr/scripts/
   ```

---

## Slow Performance

### Issue: Pack compilation takes too long

**Symptoms:**
- Context packs take >1 second
- Timeouts during pack compilation

**Diagnosis:**

```bash
# Check pack timing
icr metrics --filter pack --last 1h
```

**Solutions:**

1. **Reduce token budget**:
   ```yaml
   # In config.yaml
   pack:
     default_budget_tokens: 4000  # Lower from 8000
   ```

2. **Reduce candidate count**:
   ```yaml
   retrieval:
     initial_candidates: 50  # Lower from 100
     final_results: 15  # Lower from 20
   ```

3. **Check index size**:
   ```bash
   icr stats
   # If chunks > 100k, consider reducing scope
   ```

### Issue: Embedding generation is slow

**Symptoms:**
- First query after restart is slow
- Query latency >100ms consistently

**Diagnosis:**

```bash
# Check embedding backend
icr doctor | grep -i embedding

# Time embedding directly
time python -c "
from icd.indexing.embedder import get_embedder
e = get_embedder()
e.embed(['test query'])
"
```

**Solutions:**

1. **Ensure ONNX backend** (fastest):
   ```yaml
   embedding:
     backend: local_onnx
   ```

2. **Pre-load model** (avoid cold start):
   ```bash
   # Keep daemon running
   icr daemon --foreground
   ```

3. **Check CPU usage**:
   ```bash
   # If CPU is maxed, reduce batch size
   embedding:
     batch_size: 16  # Lower from 32
   ```

### Issue: Search queries are slow

**Symptoms:**
- env_search takes >100ms
- Hybrid search timeouts

**Diagnosis:**

```bash
# Check index health
icr index check

# Check HNSW stats
icr stats --verbose | grep -i hnsw
```

**Solutions:**

1. **Rebuild HNSW index**:
   ```bash
   icr index rebuild --vectors-only
   ```

2. **Reduce ef_search** (trade accuracy for speed):
   ```yaml
   storage:
     hnsw_ef_search: 50  # Lower from 100
   ```

3. **Enable SQLite WAL mode**:
   ```yaml
   storage:
     wal_mode: true
   ```

### Issue: File watching is slow/unresponsive

**Symptoms:**
- Changes not reflected in search
- High CPU during file changes

**Solutions:**

1. **Increase debounce delay**:
   ```yaml
   watcher:
     debounce_ms: 1000  # Increase from 500
   ```

2. **Add more ignore patterns**:
   ```yaml
   watcher:
     ignore_patterns:
       - "**/generated/**"
       - "**/tmp/**"
   ```

3. **Check for watch limit** (Linux):
   ```bash
   cat /proc/sys/fs/inotify/max_user_watches
   # If low, increase:
   echo 524288 | sudo tee /proc/sys/fs/inotify/max_user_watches
   ```

---

## Index Corruption Recovery

### Issue: SQLite database errors

**Symptoms:**
```
sqlite3.DatabaseError: database disk image is malformed
```

**Solutions:**

1. **Check disk space**:
   ```bash
   df -h ~/.icr/
   ```

2. **Attempt recovery**:
   ```bash
   # Backup existing database
   cp ~/.icr/repos/<repo_id>/index.db ~/.icr/repos/<repo_id>/index.db.bak

   # Try SQLite recovery
   sqlite3 ~/.icr/repos/<repo_id>/index.db "PRAGMA integrity_check;"

   # If corrupt, dump and restore
   sqlite3 ~/.icr/repos/<repo_id>/index.db.bak ".dump" | sqlite3 ~/.icr/repos/<repo_id>/index.db.new
   mv ~/.icr/repos/<repo_id>/index.db.new ~/.icr/repos/<repo_id>/index.db
   ```

3. **Full reindex**:
   ```bash
   icr index rebuild --full
   ```

### Issue: Vector index corrupt

**Symptoms:**
```
RuntimeError: HNSW index corrupted
```

**Solutions:**

1. **Rebuild vectors only**:
   ```bash
   icr index rebuild --vectors-only
   ```

2. **Delete and recreate**:
   ```bash
   rm ~/.icr/repos/<repo_id>/vectors.hnsw
   icr index rebuild
   ```

### Issue: Inconsistent state after crash

**Solutions:**

```bash
# Run consistency check
icr index check --repair

# If issues found, rebuild
icr index rebuild --incremental

# For severe issues, full rebuild
icr index rebuild --full
```

### Issue: Index too large

**Symptoms:**
- Disk space exhausted
- Memory errors during operation

**Solutions:**

1. **Check index size**:
   ```bash
   du -sh ~/.icr/repos/*/
   ```

2. **Reduce scope**:
   ```yaml
   watcher:
     max_file_size_kb: 200  # Lower limit
     ignore_patterns:
       - "**/test_data/**"
       - "**/fixtures/**"
   ```

3. **Clean old indexes**:
   ```bash
   # Remove indexes for deleted repos
   icr index cleanup --dry-run
   icr index cleanup
   ```

---

## Debug Logging

### Enable Debug Mode

```bash
# Via environment variable
export ICD_LOGGING__LEVEL=DEBUG
icr daemon --foreground

# Or in config
logging:
  level: DEBUG
  file_path: ~/.icr/debug.log
```

### View Specific Subsystem Logs

```bash
# Indexing
icr logs --filter indexing --level debug

# Retrieval
icr logs --filter retrieval --level debug

# Pack compilation
icr logs --filter pack --level debug
```

### Trace a Single Request

```bash
# Enable request tracing
export ICR_TRACE=1

# Run command
/ic pack "test query"

# Check trace output
cat ~/.icr/traces/latest.json | jq
```

### Profile Performance

```bash
# Enable profiling
icr daemon --profile

# After running queries, check profile
icr profile report
```

---

## Common Error Messages

### "Repository not indexed"

**Cause:** No index exists for the current directory.

**Fix:**
```bash
cd /path/to/your/project
icr index
```

### "Embedding model not found"

**Cause:** ONNX model file missing or corrupted.

**Fix:**
```bash
# Re-download model
rm -rf ~/.icr/models/
icr init --models-only
```

### "Token budget exceeded"

**Cause:** Query returned too much content.

**Fix:**
```bash
# Use smaller budget
/ic pack "query" --budget 2000

# Or configure lower default
pack:
  default_budget_tokens: 4000
```

### "Timeout during retrieval"

**Cause:** Search took too long (>5s default).

**Fix:**
```yaml
# Reduce search scope
retrieval:
  initial_candidates: 50
  final_results: 10
```

### "MCP connection refused"

**Cause:** MCP server not running or misconfigured.

**Fix:**
```bash
# Check MCP server status
icr mcp-serve --test

# Reconfigure
icr configure claude-code --force
```

### "Permission denied" accessing index

**Cause:** File permission issues.

**Fix:**
```bash
# Fix permissions
chmod -R u+rw ~/.icr/

# Check ownership
chown -R $(whoami) ~/.icr/
```

---

## Getting Help

### Gather Diagnostic Information

```bash
# Generate diagnostic report
icr diagnostics > icr-diagnostic-report.txt

# This includes:
# - Version information
# - Configuration (with secrets redacted)
# - Health check results
# - Recent errors
# - System information
```

### Where to Get Help

1. **Documentation**: Check other docs in this folder
2. **GitHub Issues**: https://github.com/icr/icr/issues
3. **Discussions**: https://github.com/icr/icr/discussions

### When Reporting Issues

Include:
- ICR version (`icr --version`)
- Python version (`python --version`)
- Operating system
- Diagnostic report (`icr diagnostics`)
- Steps to reproduce
- Expected vs actual behavior

---

## Next Steps

- [CONFIGURATION.md](CONFIGURATION.md): Configuration options
- [API_REFERENCE.md](API_REFERENCE.md): Tool documentation
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md): Contributing fixes
