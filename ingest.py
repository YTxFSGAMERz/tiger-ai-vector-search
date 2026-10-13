"""
Tiger Cloud AI-Powered Vector Search
Data Ingestion Pipeline: Seeds realistic time-series incident telemetry
with high-dimensional 384-d vector embeddings into TimescaleDB Hypertables.
"""

import os
import json
import uuid
import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

from embedder import embed_text

# Load environment variables
load_dotenv()
TIMESCALE_URL = os.getenv("TIMESCALE_SERVICE_URL")

# Curated real-world incident templates across diverse domains
INCIDENT_TEMPLATES = [
    {
        "category": "database_crash",
        "service": "order-postgres",
        "severity": "CRITICAL",
        "title": "PostgreSQL Connection Pool Exhaustion on Master Replica",
        "summary": "Application microservices failed to acquire database connections. Max client connections (500) reached, causing connection timeouts and order checkout failure.",
        "root_cause": "PgBouncer pooler connection leak triggered by unhandled transaction aborts in payment-checkout service during flash sale.",
        "resolution": "Reset leaking idle connections, increased PgBouncer max client pool, and deployed hotfix with strict transaction timeouts.",
        "telemetry": {"active_connections": 502, "pool_utilization_pct": 100.0, "p99_query_latency_ms": 8450.0, "5xx_error_rate_pct": 68.4}
    },
    {
        "category": "database_crash",
        "service": "catalog-db",
        "severity": "HIGH",
        "title": "Deadlock Detected on Product Inventory Update Table",
        "summary": "Concurrent bulk catalog inventory syncs produced row-level transaction deadlocks in PostgreSQL, rolling back stock updates.",
        "root_cause": "Multiple worker tasks updating product inventory SKUs in conflicting alphabetical lock order without consistent lock acquisition hierarchy.",
        "resolution": "Enforced deterministic SKU ID sorting prior to batch SQL UPDATEs and enabled lock_timeout=5s.",
        "telemetry": {"deadlocks_per_min": 24, "rolled_back_txns": 182, "p99_query_latency_ms": 3100.0, "5xx_error_rate_pct": 14.2}
    },
    {
        "category": "memory_exhaustion",
        "service": "recommendation-engine",
        "severity": "CRITICAL",
        "title": "Kubernetes Pod OOMKilled via High JVM OldGen Heap Leak",
        "summary": "Node agent terminated recommendation pods due to cgroup memory limit exceeded (16GB). Pod entered CrashLoopBackOff.",
        "root_cause": "Unbounded in-memory LRU cache storing vector candidate embeddings without eviction policy or weak references.",
        "resolution": "Configured Caffeine cache with strict maximumWeight and off-heap memory pagination, increased container memory ceiling to 24GB.",
        "telemetry": {"memory_allocated_mb": 16384, "heap_utilization_pct": 99.8, "oom_kills": 8, "pod_restarts": 12}
    },
    {
        "category": "memory_exhaustion",
        "service": "session-redis",
        "severity": "HIGH",
        "title": "Redis Memory Saturation and Eviction Throttling",
        "summary": "Redis cluster maxmemory-policy volatile-lru caused heavy latency degradation as keys were continuously evicted under high write pressure.",
        "root_cause": "User session tokens created without explicit TTL expiration during authentication surge, filling 32GB RAM.",
        "resolution": "Backfilled missing key TTLs with a background pipeline and scaled Redis nodes to cluster mode with 64GB memory.",
        "telemetry": {"used_memory_mb": 32600, "evicted_keys_per_sec": 4500, "latency_ms": 420.0, "hit_ratio_pct": 62.1}
    },
    {
        "category": "cpu_saturation",
        "service": "crypto-signer",
        "severity": "HIGH",
        "title": "CPU Core Saturation via Cryptographic Key Derivation",
        "summary": "All 16 vCPU cores pinned at 100% utilization. API request queues backed up to 15,000 pending requests.",
        "root_cause": "Bcrypt computational cost parameter mistakenly increased from 10 to 16 in runtime configuration deployment.",
        "resolution": "Reverted Argon2/Bcrypt work factor to baseline and added horizontal worker pod autoscale triggers.",
        "telemetry": {"cpu_utilization_pct": 99.9, "run_queue_length": 64, "pending_requests": 15200, "p99_latency_ms": 12400.0}
    },
    {
        "category": "cpu_saturation",
        "service": "analytics-worker",
        "severity": "MEDIUM",
        "title": "Infinite Loop in JSON Recursive Schema Validator",
        "summary": "Background Celery workers pegged CPU at 100% processing circular JSON schema references in user event uploads.",
        "root_cause": "Missing recursion depth limiter in recursive validation function for deeply nested customer payload metadata.",
        "resolution": "Added max_depth=10 recursion guard and isolated parser into sandboxed worker subprocess.",
        "telemetry": {"cpu_utilization_pct": 98.2, "blocked_tasks": 340, "p99_latency_ms": 7800.0}
    },
    {
        "category": "network_gateway",
        "service": "ingress-nginx",
        "severity": "CRITICAL",
        "title": "HTTP 502 Bad Gateway Spike on Public Ingress Edge",
        "summary": "External customers experienced 502 Bad Gateway responses across all API endpoints during evening peak traffic.",
        "root_cause": "Keepalive connection reuse exhausted backend socket file descriptors; NGINX upstream buffer overflow.",
        "resolution": "Tuned sysctl fs.file-max, increased worker_connections to 65535, and enabled keepalive 128 on upstream pools.",
        "telemetry": {"502_error_count": 8940, "5xx_error_rate_pct": 46.8, "active_connections": 54000, "p99_latency_ms": 5200.0}
    },
    {
        "category": "network_gateway",
        "service": "core-dns",
        "severity": "CRITICAL",
        "title": "CoreDNS Lookup Latency Degradation and Resolution Timeouts",
        "summary": "Internal service discovery DNS queries timed out across Kubernetes cluster, breaking inter-service gRPC communications.",
        "root_cause": "Conntrack table exhaustion on worker nodes due to high-rate short-lived UDP DNS queries without local caching.",
        "resolution": "Deployed NodeLocal DNSCache DaemonSet and expanded Linux conntrack max entries from 262k to 1M.",
        "telemetry": {"dns_timeout_rate_pct": 31.4, "conntrack_entries": 262140, "p99_dns_lookup_ms": 2800.0}
    },
    {
        "category": "storage_io",
        "service": "wal-archiver",
        "severity": "HIGH",
        "title": "Disk I/O IOPS Saturation During Timescale Hypertable Compaction",
        "summary": "NVMe EBS volume reached 10,000 IOPS burst limit. Write throughput throttled to 25 MB/s, delaying transaction commits.",
        "root_cause": "Scheduled chunk compression policy overlapped with intensive bulk analytical continuous aggregate backfill.",
        "resolution": "Staggered background job schedules and switched storage tier to provisioned GP3 with 16,000 baseline IOPS.",
        "telemetry": {"iops_utilized": 10000, "io_wait_pct": 74.2, "write_throughput_mb_s": 24.8, "commit_delay_ms": 860.0}
    },
    {
        "category": "storage_io",
        "service": "log-aggregator",
        "severity": "HIGH",
        "title": "Root Filesystem Capacity Warning (94% Disk Full)",
        "summary": "Elasticsearch / Clickhouse node storage volume filled up rapidly due to debug-level logging in auth microservice.",
        "root_cause": "Unfiltered stack traces logged at DEBUG level during third-party payment vendor outage, generating 40GB logs/hour.",
        "resolution": "Enforced rate-limited logging, changed default log level to INFO, and purged old uncompressed log chunks.",
        "telemetry": {"disk_used_pct": 94.6, "log_write_rate_mb_s": 120.4, "available_disk_gb": 12.4}
    },
    {
        "category": "kubernetes_cluster",
        "service": "k8s-control-plane",
        "severity": "CRITICAL",
        "title": "Etcd Leader Election Flapping and Cluster API Throttling",
        "summary": "Kubernetes API server became unresponsive. kubectl commands failed with connection refused.",
        "root_cause": "Etcd disk fsync latency spiked above 100ms because etcd shared root volume with container runtime logs.",
        "resolution": "Migrated etcd data directory to dedicated ultra-low latency NVMe mount and tuned heartbeat intervals.",
        "telemetry": {"etcd_fsync_latency_ms": 115.0, "leader_changes": 7, "api_500_rate_pct": 52.0}
    },
    {
        "category": "security_auth",
        "service": "auth-gateway",
        "severity": "HIGH",
        "title": "JWT Signing Key Rotation Failure Causing Mass 401 Unauthorized",
        "summary": "Mobile and web client requests rejected with 401 Unauthorized across all authenticated routes.",
        "root_cause": "JWKS public key cache failed to refresh from KMS endpoint due to outbound NAT gateway routing table misconfiguration.",
        "resolution": "Restored VPC route table for private subnet NAT gateway and flushed public key caching daemon.",
        "telemetry": {"401_error_rate_pct": 88.5, "rejected_requests": 34200, "jwks_fetch_failures": 140}
    }
]


def generate_time_series_incidents(days_span: int = 14, repetitions: int = 4) -> List[Dict[str, Any]]:
    """
    Generates a rich time-series dataset of incidents spanning multiple days
    with varied timestamps to test Timescale hypertable chunk partitioning.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    dataset = []

    time_offsets_hours = [
        0.5, 2.0, 5.0, 12.0, 20.0,
        28.0, 36.0, 48.0, 60.0, 72.0,
        96.0, 120.0, 168.0, 240.0, 312.0
    ]

    counter = 1
    for rep in range(repetitions):
        for idx, tmpl in enumerate(INCIDENT_TEMPLATES):
            offset_hrs = time_offsets_hours[(idx + rep * 3) % len(time_offsets_hours)] + (rep * 48.0)
            incident_time = now - datetime.timedelta(hours=offset_hrs)

            incident_id = f"INC-{incident_time.strftime('%Y%m%d')}-{counter:04d}"
            counter += 1

            # Build full contextual semantic text for embedding
            semantic_text = (
                f"{tmpl['title']}. {tmpl['summary']} "
                f"Root cause: {tmpl['root_cause']} "
                f"Resolution: {tmpl['resolution']} "
                f"Service: {tmpl['service']} Category: {tmpl['category']}"
            )

            # Generate high-dimensional 384-d vector embedding
            embedding = embed_text(semantic_text)

            dataset.append({
                "incident_time": incident_time,
                "incident_id": incident_id,
                "service_name": tmpl["service"],
                "environment": "production" if rep % 3 != 0 else "staging",
                "severity": tmpl["severity"],
                "category": tmpl["category"],
                "title": tmpl["title"],
                "summary": tmpl["summary"],
                "root_cause": tmpl["root_cause"],
                "resolution": tmpl["resolution"],
                "telemetry": tmpl["telemetry"],
                "embedding": embedding
            })

    return dataset


def ingest_data():
    if not TIMESCALE_URL:
        raise ValueError("TIMESCALE_SERVICE_URL environment variable is missing!")

    print(f" Connecting to Tiger Cloud database...")
    conn = psycopg2.connect(TIMESCALE_URL)
    register_vector(conn)
    cursor = conn.cursor()

    print(" Generating synthetic DevOps incidents and 384-dimensional vector embeddings...")
    incidents = generate_time_series_incidents(days_span=21, repetitions=4)
    print(f" Generated {len(incidents)} realistic incident records with vector embeddings.")

    insert_query = """
    INSERT INTO incident_logs (
        incident_time, incident_id, service_name, environment,
        severity, category, title, summary, root_cause, resolution,
        telemetry, embedding
    ) VALUES %s
    ON CONFLICT (incident_time, incident_id) DO UPDATE SET
        summary = EXCLUDED.summary,
        embedding = EXCLUDED.embedding,
        telemetry = EXCLUDED.telemetry;
    """

    records = [
        (
            inc["incident_time"],
            inc["incident_id"],
            inc["service_name"],
            inc["environment"],
            inc["severity"],
            inc["category"],
            inc["title"],
            inc["summary"],
            inc["root_cause"],
            inc["resolution"],
            json.dumps(inc["telemetry"]),
            inc["embedding"]
        )
        for inc in incidents
    ]

    print(" Ingesting records into TimescaleDB Hypertable...")
    execute_values(cursor, insert_query, records, template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)")
    conn.commit()

    # Query hypertable chunk statistics
    cursor.execute("""
        SELECT count(*) AS total_incidents,
               min(incident_time) AS earliest_record,
               max(incident_time) AS latest_record
        FROM incident_logs;
    """)
    stats = cursor.fetchone()

    cursor.execute("""
        SELECT chunk_name, range_start, range_end
        FROM timescaledb_information.chunks
        WHERE hypertable_name = 'incident_logs'
        ORDER BY range_start DESC;
    """)
    chunks = cursor.fetchall()

    print("\n Ingestion Completed Successfully!")
    print(f"   Total Records Stored: {stats[0]}")
    print(f"   Time Horizon: {stats[1]} to {stats[2]}")
    print(f"   Timescale Chunks Created: {len(chunks)}")
    for ch in chunks[:5]:
        print(f"     - Chunk: {ch[0]} | Range: {ch[1]} -> {ch[2]}")
    if len(chunks) > 5:
        print(f"     ... and {len(chunks) - 5} more chunks partitioned by timestamp.")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    ingest_data()
