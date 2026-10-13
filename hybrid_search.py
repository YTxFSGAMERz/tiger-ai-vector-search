"""
Tiger Cloud AI-Powered Vector Search & Hybrid RAG Retrieval Engine
Demonstrates simultaneous semantic vector similarity search,
Timescale hypertable chunk pruning, and metadata filtering.
"""

import os
import sys
import json
import argparse
import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector

from embedder import embed_text

load_dotenv()
TIMESCALE_URL = os.getenv("TIMESCALE_SERVICE_URL")


class TigerVectorSearch:
    def __init__(self, connection_url: Optional[str] = None):
        self.url = connection_url or TIMESCALE_URL
        if not self.url:
            raise ValueError("TIMESCALE_SERVICE_URL environment variable is missing.")

    def get_connection(self):
        conn = psycopg2.connect(self.url)
        register_vector(conn)
        return conn

    def search_hybrid(
        self,
        query_text: str,
        days_back: Optional[int] = 30,
        service: Optional[str] = None,
        severity: Optional[str] = None,
        top_k: int = 5,
        decay_rate: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Executes a hybrid similarity search combining:
        1. 384-dimensional dense vector cosine distance (<=>)
        2. TimescaleDB hypertable partition pruning via time filter
        3. Relational metadata filters (service, severity)
        4. Temporal recency decay scoring
        """
        query_vector = embed_text(query_text)

        now = datetime.datetime.now(datetime.timezone.utc)
        start_time = now - datetime.timedelta(days=days_back) if days_back else now - datetime.timedelta(days=365)

        sql = """
        SELECT
            incident_id,
            incident_time,
            service_name,
            severity,
            category,
            title,
            summary,
            root_cause,
            resolution,
            telemetry,
            cosine_distance,
            semantic_similarity,
            temporal_score,
            hybrid_score
        FROM search_similar_incidents(
            query_vector => %s::vector,
            start_time => %s::timestamptz,
            end_time => %s::timestamptz,
            target_service => %s::varchar,
            target_severity => %s::varchar,
            match_limit => %s::int,
            decay_rate => %s::float8
        );
        """

        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            sql,
            (query_vector, start_time, now, service, severity, top_k, decay_rate)
        )
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        results = [dict(zip(columns, row)) for row in rows]
        cursor.close()
        conn.close()
        return results

    def explain_vector_query(self, query_text: str, days_back: int = 7) -> str:
        """
        Runs EXPLAIN ANALYZE to illustrate how TimescaleDB prunes chunks
        and leverages the vector index (HNSW / DiskANN).
        """
        query_vector = embed_text(query_text)
        now = datetime.datetime.now(datetime.timezone.utc)
        start_time = now - datetime.timedelta(days=days_back)

        sql = """
        EXPLAIN (ANALYZE, BUFFERS)
        SELECT incident_id, incident_time, service_name, title,
               (embedding <=> %s::vector) AS cosine_dist
        FROM incident_logs
        WHERE incident_time >= %s AND incident_time <= %s
        ORDER BY embedding <=> %s::vector
        LIMIT 5;
        """

        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(sql, (query_vector, start_time, now, query_vector))
        lines = [r[0] for r in cursor.fetchall()]
        cursor.close()
        conn.close()
        return "\n".join(lines)

    def generate_rag_context(self, query_text: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved vector search results into an LLM-ready prompt context.
        """
        header = f"###  Tiger Cloud Hybrid RAG Context\n**User Issue / Incident Query:** `{query_text}`\n\n"
        context_items = []
        for idx, res in enumerate(search_results, 1):
            time_str = res["incident_time"].strftime("%Y-%m-%d %H:%M UTC")
            sim_pct = f"{res['semantic_similarity'] * 100:.1f}%"
            hybrid_pct = f"{res['hybrid_score'] * 100:.1f}%"
            item = (
                f"**Match #{idx}: {res['title']}** (ID: `{res['incident_id']}`)\n"
                f"- **Time:** {time_str} | **Service:** `{res['service_name']}` | **Severity:** `{res['severity']}`\n"
                f"- **Semantic Similarity:** {sim_pct} | **Hybrid Recency Score:** {hybrid_pct}\n"
                f"- **Summary:** {res['summary']}\n"
                f"- **Root Cause:** {res['root_cause']}\n"
                f"- **Resolution:** {res['resolution']}\n"
                f"- **Telemetry Context:** `{json.dumps(res['telemetry'])}`\n"
            )
            context_items.append(item)
        return header + "\n---\n".join(context_items)


def main():
    parser = argparse.ArgumentParser(description="Tiger Cloud pgvector + Timescale Hybrid Search")
    parser.add_argument("--query", type=str, default="PostgreSQL connection pool exhausted, queries timing out under heavy traffic", help="Search query")
    parser.add_argument("--days", type=int, default=14, help="Time horizon in days for hypertable chunk pruning")
    parser.add_argument("--service", type=str, default=None, help="Optional service name filter")
    parser.add_argument("--severity", type=str, default=None, help="Optional severity filter (CRITICAL, HIGH, MEDIUM)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of nearest neighbors")
    parser.add_argument("--explain", action="store_true", help="Print EXPLAIN ANALYZE execution plan")

    args = parser.parse_args()

    searcher = TigerVectorSearch()

    print("\n" + "=" * 80)
    print(f" TIGER CLOUD AI-POWERED VECTOR SEARCH (pgvector + pgvectorscale)")
    print("=" * 80)
    print(f" Query: \"{args.query}\"")
    print(f" Time Range: Last {args.days} days (Timescale hypertable chunk filtering)")
    if args.service:
        print(f" Service Filter: {args.service}")
    if args.severity:
        print(f" Severity Filter: {args.severity}")
    print(f" Metric: Cosine Distance (<=>) with Recency Decay Ranking")
    print("-" * 80)

    results = searcher.search_hybrid(
        query_text=args.query,
        days_back=args.days,
        service=args.service,
        severity=args.severity,
        top_k=args.top_k
    )

    for idx, r in enumerate(results, 1):
        print(f"\n[Match #{idx}] {r['title']}")
        print(f"  Incident ID   : {r['incident_id']}")
        print(f"  Timestamp     : {r['incident_time']}")
        print(f"  Service/Class : {r['service_name']} ({r['category']}) | Severity: {r['severity']}")
        print(f"  Cosine Dist   : {r['cosine_distance']:.4f}")
        print(f"  Similarity    : {r['semantic_similarity'] * 100:.2f}%")
        print(f"  Temporal Decay: {r['temporal_score']:.4f}")
        print(f"  Hybrid Score  : {r['hybrid_score'] * 100:.2f}%")
        print(f"  Summary       : {r['summary'][:120]}...")
        print(f"  Resolution    : {r['resolution'][:120]}...")

    print("\n" + "=" * 80)
    print(" GENERATED RETRIEVAL-AUGMENTED GENERATION (RAG) PROMPT CONTEXT")
    print("=" * 80)
    rag_context = searcher.generate_rag_context(args.query, results)
    print(rag_context)

    if args.explain:
        print("\n" + "=" * 80)
        print(" EXPLAIN ANALYZE PLAN (Hypertable Pruning + Vector Execution)")
        print("=" * 80)
        plan = searcher.explain_vector_query(args.query, days_back=args.days)
        print(plan)


if __name__ == "__main__":
    main()
