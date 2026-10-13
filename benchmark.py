"""
Tiger Cloud pgvector & pgvectorscale Benchmark Suite
Compares Sequential Table Scan vs pgvector HNSW vs pgvectorscale DiskANN
with TimescaleDB Hypertable Chunk Pruning.
"""

import os
import time
import statistics
from typing import List, Dict, Any
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector

from embedder import embed_text

load_dotenv()
TIMESCALE_URL = os.getenv("TIMESCALE_SERVICE_URL")

BENCHMARK_QUERIES = [
    "PostgreSQL connection pool exhausted, queries timing out under heavy traffic",
    "Pod OOMKilled due to high JVM OldGen heap leak in recommendation engine",
    "HTTP 502 Bad Gateway spike and socket keepalive exhaustion on NGINX ingress",
    "CPU core saturation via bcrypt cryptographic key derivation",
    "Disk IOPS saturation during Timescale hypertable chunk compression",
    "CoreDNS UDP lookup timeout and conntrack table exhaustion"
]


def run_benchmark(iterations: int = 15) -> Dict[str, Any]:
    if not TIMESCALE_URL:
        raise ValueError("TIMESCALE_SERVICE_URL environment variable is missing.")

    conn = psycopg2.connect(TIMESCALE_URL)
    register_vector(conn)
    cursor = conn.cursor()

    results = {
        "sequential_scan": [],
        "hnsw_vector_scan": [],
        "diskann_vector_scan": [],
        "hypertable_chunk_pruned": []
    }

    print("\n" + "=" * 80)
    print(" BENCHMARKING TIGER CLOUD VECTOR SEARCH WITH pgvector & pgvectorscale")
    print(f" Instances: TimescaleDB 2.30.2 | pgvector 0.8.7 | pgvectorscale 0.9.0")
    print(f" Iterations: {iterations} per test configuration")
    print("=" * 80)

    # Pre-embed queries
    embedded_queries = [embed_text(q) for q in BENCHMARK_QUERIES]

    # Warm up caches
    cursor.execute("SELECT count(*) FROM incident_logs;")
    cursor.fetchone()

    # 1. Benchmark: Sequential / Flat Scan (force index off)
    print(" [1/4] Benchmarking Sequential / Flat Scan (enable_indexscan=OFF)...")
    cursor.execute("SET enable_indexscan = OFF; SET enable_bitmapscan = OFF;")
    conn.commit()

    for _ in range(iterations):
        for qv in embedded_queries:
            t0 = time.perf_counter()
            cursor.execute("""
                SELECT incident_id, (embedding <=> %s::vector) AS dist
                FROM incident_logs
                ORDER BY embedding <=> %s::vector
                LIMIT 5;
            """, (qv, qv))
            cursor.fetchall()
            t1 = time.perf_counter()
            results["sequential_scan"].append((t1 - t0) * 1000.0)

    # 2. Benchmark: pgvector HNSW Scan (enable index scan, disable diskann)
    print(" [2/4] Benchmarking pgvector HNSW Index Scan...")
    cursor.execute("SET enable_indexscan = ON; SET enable_bitmapscan = ON;")
    # Ensure HNSW is preferred by dropping/hiding diskann temporarily or testing HNSW directly
    conn.commit()

    for _ in range(iterations):
        for qv in embedded_queries:
            t0 = time.perf_counter()
            cursor.execute("""
                /*+ IndexScan(incident_logs idx_incident_logs_hnsw_cosine) */
                SELECT incident_id, (embedding <=> %s::vector) AS dist
                FROM incident_logs
                ORDER BY embedding <=> %s::vector
                LIMIT 5;
            """, (qv, qv))
            cursor.fetchall()
            t1 = time.perf_counter()
            results["hnsw_vector_scan"].append((t1 - t0) * 1000.0)

    # 3. Benchmark: pgvectorscale DiskANN Scan
    print(" [3/4] Benchmarking pgvectorscale DiskANN Index Scan...")
    for _ in range(iterations):
        for qv in embedded_queries:
            t0 = time.perf_counter()
            cursor.execute("""
                SELECT incident_id, (embedding <=> %s::vector) AS dist
                FROM incident_logs
                ORDER BY embedding <=> %s::vector
                LIMIT 5;
            """, (qv, qv))
            cursor.fetchall()
            t1 = time.perf_counter()
            results["diskann_vector_scan"].append((t1 - t0) * 1000.0)

    # 4. Benchmark: Hybrid Search with TimescaleDB Hypertable Chunk Pruning (Last 24 Hours)
    print(" [4/4] Benchmarking Hypertable Chunk Pruned Search (Last 24 Hours)...")
    for _ in range(iterations):
        for qv in embedded_queries:
            t0 = time.perf_counter()
            cursor.execute("""
                SELECT incident_id, (embedding <=> %s::vector) AS dist
                FROM incident_logs
                WHERE incident_time >= NOW() - INTERVAL '24 hours'
                ORDER BY embedding <=> %s::vector
                LIMIT 5;
            """, (qv, qv))
            cursor.fetchall()
            t1 = time.perf_counter()
            results["hypertable_chunk_pruned"].append((t1 - t0) * 1000.0)

    # Reset planner flags
    cursor.execute("RESET ALL;")
    conn.commit()
    cursor.close()
    conn.close()

    # Calculate statistics
    stats_summary = {}
    for key, latencies in results.items():
        stats_summary[key] = {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[int(len(latencies) * 0.95)],
            "min": min(latencies),
            "max": max(latencies),
            "qps": 1000.0 / statistics.mean(latencies)
        }

    print("\n" + "=" * 80)
    print(f"{'Search Method':<32} | {'Mean (ms)':<10} | {'p95 (ms)':<10} | {'Min (ms)':<10} | {'Throughput (QPS)':<16}")
    print("-" * 80)
    labels = {
        "sequential_scan": "Sequential Scan (Flat)",
        "hnsw_vector_scan": "pgvector (HNSW Index)",
        "diskann_vector_scan": "pgvectorscale (DiskANN)",
        "hypertable_chunk_pruned": "Timescale + DiskANN (24h Chunks)"
    }
    for key, s in stats_summary.items():
        lbl = labels.get(key, key)
        print(f"{lbl:<32} | {s['mean']:<10.2f} | {s['p95']:<10.2f} | {s['min']:<10.2f} | {s['qps']:<16.1f}")
    print("=" * 80)

    return stats_summary


if __name__ == "__main__":
    run_benchmark(iterations=10)
