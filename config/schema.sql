-- ICR Database Schema
-- SQLite with FTS5 for lexical search
--
-- This schema defines the storage structure for the Context Environment (E).
-- Each repository has its own database at: ~/.icr/repos/<repo_id>/icr.sqlite

-- =============================================================================
-- Core Tables
-- =============================================================================

-- Repository metadata
CREATE TABLE IF NOT EXISTS repo_meta (
    id INTEGER PRIMARY KEY,
    repo_id TEXT UNIQUE NOT NULL,
    repo_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_indexed_at TEXT,
    config_hash TEXT,

    -- Statistics
    file_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    vector_count INTEGER DEFAULT 0,
    contract_count INTEGER DEFAULT 0
);

-- Indexed files
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    relative_path TEXT NOT NULL,
    language TEXT,
    file_hash TEXT NOT NULL,
    mtime TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    indexed_at TEXT NOT NULL DEFAULT (datetime('now')),

    -- File classification
    is_contract BOOLEAN DEFAULT FALSE,
    is_test BOOLEAN DEFAULT FALSE,
    is_config BOOLEAN DEFAULT FALSE,

    -- Indexes
    UNIQUE(path)
);
CREATE INDEX IF NOT EXISTS idx_files_language ON files(language);
CREATE INDEX IF NOT EXISTS idx_files_is_contract ON files(is_contract);
CREATE INDEX IF NOT EXISTS idx_files_mtime ON files(mtime);

-- Chunks (semantic units extracted from files)
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,  -- Content-hash based stable ID
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,

    -- Location
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    start_offset INTEGER,
    end_offset INTEGER,

    -- Content
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,

    -- Symbol information
    symbol_type TEXT,  -- function, class, method, module, etc.
    symbol_name TEXT,
    symbol_path TEXT,  -- Fully qualified path: module.Class.method

    -- Metadata
    language TEXT,
    has_docstring BOOLEAN DEFAULT FALSE,
    indexed_at TEXT NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_symbol_name ON chunks(symbol_name);
CREATE INDEX IF NOT EXISTS idx_chunks_symbol_type ON chunks(symbol_type);

-- FTS5 virtual table for lexical search
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    symbol_name,
    symbol_path,
    content=chunks,
    content_rowid=id,
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content, symbol_name, symbol_path)
    VALUES (new.id, new.content, new.symbol_name, new.symbol_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, symbol_name, symbol_path)
    VALUES ('delete', old.id, old.content, old.symbol_name, old.symbol_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, symbol_name, symbol_path)
    VALUES ('delete', old.id, old.content, old.symbol_name, old.symbol_path);
    INSERT INTO chunks_fts(rowid, content, symbol_name, symbol_path)
    VALUES (new.id, new.content, new.symbol_name, new.symbol_path);
END;

-- =============================================================================
-- Vector Storage Metadata
-- =============================================================================

-- Vector embeddings metadata (actual vectors stored in separate HNSW index)
CREATE TABLE IF NOT EXISTS vectors (
    id INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    vector_id INTEGER NOT NULL,  -- ID in HNSW index
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    dtype TEXT NOT NULL DEFAULT 'float16',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),

    UNIQUE(chunk_id, model)
);
CREATE INDEX IF NOT EXISTS idx_vectors_chunk ON vectors(chunk_id);
CREATE INDEX IF NOT EXISTS idx_vectors_vector_id ON vectors(vector_id);

-- =============================================================================
-- Contract Index
-- =============================================================================

-- Detected contracts (API definitions, schemas, etc.)
CREATE TABLE IF NOT EXISTS contracts (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    contract_type TEXT NOT NULL,  -- openapi, protobuf, graphql, jsonschema, typescript-types

    -- Contract metadata
    name TEXT,
    version TEXT,

    -- Normalized representation
    normalized_json TEXT,

    indexed_at TEXT NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_contracts_file ON contracts(file_id);
CREATE INDEX IF NOT EXISTS idx_contracts_type ON contracts(contract_type);

-- Contract entities (endpoints, types, fields)
CREATE TABLE IF NOT EXISTS contract_entities (
    id INTEGER PRIMARY KEY,
    contract_id INTEGER NOT NULL REFERENCES contracts(id) ON DELETE CASCADE,

    -- Entity type
    entity_type TEXT NOT NULL,  -- endpoint, method, field, type, schema

    -- Entity identification
    name TEXT NOT NULL,
    path TEXT,  -- e.g., /api/users/{id} or User.email

    -- Entity details
    http_method TEXT,  -- GET, POST, etc. for endpoints
    data_type TEXT,    -- string, number, etc. for fields
    required BOOLEAN,

    -- Location in source
    start_line INTEGER,
    end_line INTEGER,

    FOREIGN KEY (contract_id) REFERENCES contracts(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_contract_entities_contract ON contract_entities(contract_id);
CREATE INDEX IF NOT EXISTS idx_contract_entities_type ON contract_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_contract_entities_name ON contract_entities(name);

-- Contract usages (references to contract entities from code)
CREATE TABLE IF NOT EXISTS contract_usages (
    id INTEGER PRIMARY KEY,
    entity_id INTEGER NOT NULL REFERENCES contract_entities(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,

    -- Usage type
    usage_type TEXT NOT NULL,  -- imports, calls, serializes, deserializes, tests

    -- Confidence score
    confidence REAL DEFAULT 1.0,

    FOREIGN KEY (entity_id) REFERENCES contract_entities(id) ON DELETE CASCADE,
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_contract_usages_entity ON contract_usages(entity_id);
CREATE INDEX IF NOT EXISTS idx_contract_usages_chunk ON contract_usages(chunk_id);

-- =============================================================================
-- Memory Storage
-- =============================================================================

-- Pinned invariants (user-defined, never inferred)
CREATE TABLE IF NOT EXISTS pinned_invariants (
    id INTEGER PRIMARY KEY,
    pin_id TEXT UNIQUE NOT NULL,

    -- Content
    content TEXT NOT NULL,

    -- Metadata
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    created_by TEXT,  -- 'user' or session_id

    -- Priority for pack inclusion
    priority INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_pins_priority ON pinned_invariants(priority DESC);

-- Decisions (extracted from ledger blocks)
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,

    -- Content
    content TEXT NOT NULL,

    -- Context
    related_files TEXT,  -- JSON array of paths
    related_chunks TEXT, -- JSON array of chunk_ids

    -- Metadata
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    work_unit_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_session ON decisions(session_id);
CREATE INDEX IF NOT EXISTS idx_decisions_work_unit ON decisions(work_unit_id);

-- Todos (extracted from ledger blocks)
CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,

    -- Content
    content TEXT NOT NULL,

    -- Status
    status TEXT DEFAULT 'pending',  -- pending, completed, cancelled
    completed_at TEXT,

    -- Metadata
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    work_unit_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_todos_session ON todos(session_id);
CREATE INDEX IF NOT EXISTS idx_todos_status ON todos(status);

-- Open questions (extracted from ledger blocks)
CREATE TABLE IF NOT EXISTS open_questions (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,

    -- Content
    content TEXT NOT NULL,

    -- Status
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TEXT,
    resolution TEXT,

    -- Metadata
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    work_unit_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_questions_session ON open_questions(session_id);
CREATE INDEX IF NOT EXISTS idx_questions_resolved ON open_questions(resolved);

-- Work units (segments based on contract touch, branch change, or idle gap)
CREATE TABLE IF NOT EXISTS work_units (
    id INTEGER PRIMARY KEY,
    unit_id TEXT UNIQUE NOT NULL,

    -- Boundaries
    started_at TEXT NOT NULL,
    ended_at TEXT,

    -- Trigger
    trigger_type TEXT,  -- branch_change, contract_touch, idle_gap, manual
    trigger_details TEXT,

    -- Summary
    files_touched TEXT,  -- JSON array
    contracts_touched TEXT,  -- JSON array
    decision_count INTEGER DEFAULT 0,
    todo_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_work_units_started ON work_units(started_at);

-- =============================================================================
-- Mode Gating and Priors
-- =============================================================================

-- Beta priors for task-class success rates
CREATE TABLE IF NOT EXISTS task_priors (
    id INTEGER PRIMARY KEY,
    task_class TEXT NOT NULL,
    mode TEXT NOT NULL,  -- pack, rlm

    -- Beta distribution parameters
    alpha REAL NOT NULL DEFAULT 2.0,
    beta REAL NOT NULL DEFAULT 2.0,

    -- Statistics
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    last_updated_at TEXT NOT NULL DEFAULT (datetime('now')),

    UNIQUE(task_class, mode)
);
CREATE INDEX IF NOT EXISTS idx_priors_task_mode ON task_priors(task_class, mode);

-- Gating decisions log
CREATE TABLE IF NOT EXISTS gating_log (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,

    -- Input signals
    prompt_hash TEXT,
    task_class TEXT,
    entropy REAL,
    contract_touched BOOLEAN,

    -- Decision
    mode_selected TEXT NOT NULL,  -- pack, rlm
    reason_codes TEXT,  -- JSON array

    -- Outcome (updated later by Stop hook)
    outcome TEXT,  -- success, failure, unknown

    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_gating_session ON gating_log(session_id);
CREATE INDEX IF NOT EXISTS idx_gating_mode ON gating_log(mode_selected);

-- =============================================================================
-- Telemetry (Local Only)
-- =============================================================================

-- Tool invocation metrics
CREATE TABLE IF NOT EXISTS tool_metrics (
    id INTEGER PRIMARY KEY,
    tool_name TEXT NOT NULL,

    -- Timing
    latency_ms REAL NOT NULL,

    -- Request details
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    response_tokens INTEGER,

    -- Outcome
    success BOOLEAN NOT NULL,
    error_code TEXT,

    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_metrics_tool ON tool_metrics(tool_name);
CREATE INDEX IF NOT EXISTS idx_metrics_created ON tool_metrics(created_at);

-- Session metrics (EWR, IMR proxies)
CREATE TABLE IF NOT EXISTS session_metrics (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,

    -- EWR proxy metrics
    exploration_tool_count INTEGER DEFAULT 0,
    exploration_output_bytes INTEGER DEFAULT 0,
    production_tool_count INTEGER DEFAULT 0,
    repeated_reads_before_edit INTEGER DEFAULT 0,

    -- IMR proxy metrics
    contract_changes INTEGER DEFAULT 0,
    related_updates INTEGER DEFAULT 0,
    missed_updates INTEGER DEFAULT 0,

    -- Computed metrics (updated at session end)
    ewr_estimate REAL,
    imr_estimate REAL,

    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_session_metrics_session ON session_metrics(session_id);

-- =============================================================================
-- Transcript Storage
-- =============================================================================

-- Ingested transcript turns
CREATE TABLE IF NOT EXISTS transcript_turns (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,

    -- Content
    role TEXT NOT NULL,  -- user, assistant
    content TEXT NOT NULL,

    -- Extracted data
    ledger_json TEXT,  -- Extracted ledger if present
    tools_used TEXT,   -- JSON array of tool names
    files_touched TEXT, -- JSON array of file paths

    -- Metadata
    timestamp TEXT NOT NULL,

    UNIQUE(session_id, turn_index)
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON transcript_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON transcript_turns(timestamp);

-- FTS for transcript search
CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
    content,
    content=transcript_turns,
    content_rowid=id,
    tokenize='porter unicode61'
);

-- Triggers for transcript FTS
CREATE TRIGGER IF NOT EXISTS transcript_ai AFTER INSERT ON transcript_turns BEGIN
    INSERT INTO transcript_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS transcript_ad AFTER DELETE ON transcript_turns BEGIN
    INSERT INTO transcript_fts(transcript_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;

-- =============================================================================
-- Diff Storage
-- =============================================================================

-- Stored diffs for change tracking
CREATE TABLE IF NOT EXISTS diffs (
    id INTEGER PRIMARY KEY,
    diff_id TEXT UNIQUE NOT NULL,

    -- Scope
    file_path TEXT NOT NULL,

    -- Content
    diff_content TEXT NOT NULL,
    added_lines INTEGER DEFAULT 0,
    removed_lines INTEGER DEFAULT 0,

    -- Metadata
    base_ref TEXT,  -- Git ref or 'working-tree'
    target_ref TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_diffs_file ON diffs(file_path);
CREATE INDEX IF NOT EXISTS idx_diffs_created ON diffs(created_at);

-- =============================================================================
-- Version and Migration
-- =============================================================================

-- Schema version for migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Initialize version if not exists
INSERT OR IGNORE INTO schema_version (version) VALUES (1);
