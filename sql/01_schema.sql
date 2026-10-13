-- ============================================================================
-- Tiger Cloud AI-Powered Vector Search with pgvector & pgvectorscale
-- Schema: Incident Intelligence & Time-Series Vector Hypertable
-- ============================================================================

-- 1. Enable Required Extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;

-- 2. Create Incident Telemetry & Embeddings Table
CREATE TABLE IF NOT EXISTS incident_logs (
    incident_time TIMESTAMPTZ NOT NULL,
    incident_id VARCHAR(64) NOT NULL,
    service_name VARCHAR(64) NOT NULL,
    environment VARCHAR(32) NOT NULL DEFAULT 'production',
    severity VARCHAR(16) NOT NULL,
    category VARCHAR(64) NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    root_cause TEXT NOT NULL,
    resolution TEXT NOT NULL,
    telemetry JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(384) NOT NULL,
    PRIMARY KEY (incident_time, incident_id)
);

-- 3. Transform into a TimescaleDB Hypertable partitioned by timestamp
SELECT create_hypertable(
    'incident_logs',
    'incident_time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- 4. Create Relational Metadata Indexes (B-tree)
CREATE INDEX IF NOT EXISTS idx_incident_logs_service_time
    ON incident_logs (service_name, incident_time DESC);

CREATE INDEX IF NOT EXISTS idx_incident_logs_severity_time
    ON incident_logs (severity, incident_time DESC);

CREATE INDEX IF NOT EXISTS idx_incident_logs_category_time
    ON incident_logs (category, incident_time DESC);

-- 5. Create pgvector HNSW Index (Hierarchical Navigable Small World) for fast Cosine Similarity
CREATE INDEX IF NOT EXISTS idx_incident_logs_hnsw_cosine
    ON incident_logs USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 6. Create pgvectorscale DiskANN Index (StreamingDiskANN) for NVMe & high-throughput scaling
CREATE INDEX IF NOT EXISTS idx_incident_logs_diskann
    ON incident_logs USING diskann (embedding);

-- 7. High-Performance Hybrid Similarity Search Function
-- Filters by time boundaries (triggering Timescale chunk pruning), metadata, and semantic vector distance
CREATE OR REPLACE FUNCTION search_similar_incidents(
    query_vector vector(384),
    start_time TIMESTAMPTZ DEFAULT NOW() - INTERVAL '30 days',
    end_time TIMESTAMPTZ DEFAULT NOW(),
    target_service VARCHAR DEFAULT NULL,
    target_severity VARCHAR DEFAULT NULL,
    match_limit INT DEFAULT 5,
    decay_rate FLOAT DEFAULT 0.05
)
RETURNS TABLE (
    incident_id VARCHAR,
    incident_time TIMESTAMPTZ,
    service_name VARCHAR,
    severity VARCHAR,
    category VARCHAR,
    title TEXT,
    summary TEXT,
    root_cause TEXT,
    resolution TEXT,
    telemetry JSONB,
    cosine_distance FLOAT,
    semantic_similarity FLOAT,
    temporal_score FLOAT,
    hybrid_score FLOAT
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        i.incident_id,
        i.incident_time,
        i.service_name,
        i.severity,
        i.category,
        i.title,
        i.summary,
        i.root_cause,
        i.resolution,
        i.telemetry,
        (i.embedding <=> query_vector)::FLOAT AS cosine_distance,
        (1.0 - (i.embedding <=> query_vector))::FLOAT AS semantic_similarity,
        (1.0 / (1.0 + decay_rate * EXTRACT(EPOCH FROM (NOW() - i.incident_time)) / 86400.0))::FLOAT AS temporal_score,
        ((1.0 - (i.embedding <=> query_vector)) * (1.0 / (1.0 + decay_rate * EXTRACT(EPOCH FROM (NOW() - i.incident_time)) / 86400.0)))::FLOAT AS hybrid_score
    FROM incident_logs i
    WHERE i.incident_time >= start_time
      AND i.incident_time <= end_time
      AND (target_service IS NULL OR i.service_name = target_service)
      AND (target_severity IS NULL OR i.severity = target_severity)
    ORDER BY (i.embedding <=> query_vector) ASC
    LIMIT match_limit;
$$;
