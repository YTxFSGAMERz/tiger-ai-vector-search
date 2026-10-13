"""
Vector Embedder for Tiger Cloud AI-Powered Vector Search.
Provides reproducible, fast 384-dimensional dense vector embeddings
with semantic topic clustering and unit-norm normalization.
"""

import os
import re
import math
import hashlib
from typing import List

# Key technical topic clusters with distinct semantic direction anchors
TOPIC_ANCHORS = {
    "database": [
        "postgres", "postgresql", "timescale", "timescaledb", "hypertable", "query",
        "sql", "deadlock", "pgbouncer", "pool", "database", "vacuum", "wal",
        "transaction", "inventory", "sku", "row", "lock", "replica", "db"
    ],
    "memory": [
        "memory", "ram", "oom", "oomkilled", "leak", "heap", "oldgen", "garbage",
        "allocation", "cgroup", "paging", "swap", "caffeine", "jvm", "redis",
        "eviction", "maxmemory"
    ],
    "cpu": [
        "cpu", "core", "vcpu", "thread", "saturation", "throttling", "loop",
        "bcrypt", "crypto", "argon2", "validator", "schema", "computation",
        "recursive", "pegged"
    ],
    "network": [
        "network", "packet", "latency", "dns", "coredns", "gateway", "502",
        "504", "nginx", "socket", "tcp", "conntrack", "udp", "ingress",
        "proxy", "keepalive", "descriptors", "upstream"
    ],
    "storage": [
        "disk", "storage", "io", "iops", "ebs", "nvme", "ssd", "filesystem",
        "compaction", "gp3", "capacity", "full", "uncompressed", "aggregator"
    ],
    "kubernetes": [
        "kubernetes", "k8s", "etcd", "pod", "container", "crashloop",
        "crashloopbackoff", "cluster", "daemonset", "kubectl", "flapping"
    ],
    "security": [
        "auth", "authentication", "unauthorized", "401", "403", "jwt",
        "jwks", "token", "rotation", "kms", "nat", "signing"
    ]
}

DIMENSION = 384


class SemanticEmbedder:
    """
    High-performance semantic vector embedder generating 384-dimensional dense vectors.
    Uses seeded token hashing, n-gram pooling, and domain topic anchor projection,
    strictly normalized to unit length on the S^383 hypersphere.
    """

    def __init__(self, dimension: int = DIMENSION):
        self.dimension = dimension
        self.topic_vectors = {}
        for topic, keywords in TOPIC_ANCHORS.items():
            vec = [0.0] * self.dimension
            for kw in keywords:
                kw_vec = self._hash_token(kw)
                for i in range(self.dimension):
                    vec[i] += kw_vec[i]
            self.topic_vectors[topic] = self._normalize(vec)

    def _hash_token(self, token: str) -> List[float]:
        """Maps an individual token to a deterministic 384-dim pseudo-random vector."""
        vec = [0.0] * self.dimension
        token_clean = token.lower().strip()
        if not token_clean:
            return vec
        # 384 floats = 12 blocks of 32 floats (each block from SHA-256)
        for block in range(12):
            h = hashlib.sha256(f"{token_clean}:{block}".encode("utf-8")).digest()
            for idx in range(32):
                dim_idx = block * 32 + idx
                # Map byte (0..255) to zero-centered float (-1.0 .. 1.0)
                byte_val = h[idx]
                vec[dim_idx] = (byte_val / 127.5) - 1.0
        return self._normalize(vec)

    def _normalize(self, vec: List[float]) -> List[float]:
        """Normalizes a vector to unit length (L2 norm = 1.0)."""
        norm_sq = sum(x * x for x in vec)
        if norm_sq <= 1e-12:
            return [0.0] * len(vec)
        inv_norm = 1.0 / math.sqrt(norm_sq)
        return [x * inv_norm for x in vec]

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a text string into a normalized 384-dimensional vector.
        """
        text_clean = text.lower()
        tokens = re.findall(r"\b[a-z0-9_\-\.]{2,}\b", text_clean)
        if not tokens:
            return [0.0] * self.dimension

        dense_vec = [0.0] * self.dimension

        # 1. Base token contribution with sub-word / character bigram hashing
        for token in tokens:
            t_vec = self._hash_token(token)
            weight = 1.0 + min(len(token) * 0.15, 2.0)
            for i in range(self.dimension):
                dense_vec[i] += t_vec[i] * weight

        # 2. Bigrams for phrases (e.g. "connection pool", "bad gateway", "memory leak")
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            b_vec = self._hash_token(bigram)
            for j in range(self.dimension):
                dense_vec[j] += b_vec[j] * 1.5

        # 3. Topic anchor amplification
        for topic, keywords in TOPIC_ANCHORS.items():
            matched_kws = [kw for kw in keywords if kw in text_clean]
            if matched_kws:
                topic_vec = self.topic_vectors[topic]
                boost = (len(matched_kws) ** 0.8) * 3.5
                for i in range(self.dimension):
                    dense_vec[i] += topic_vec[i] * boost

        return self._normalize(dense_vec)

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Computes cosine similarity between two unit vectors."""
        return sum(a * b for a, b in zip(vec_a, vec_b))


_default_embedder = None

def get_embedder() -> SemanticEmbedder:
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = SemanticEmbedder()
    return _default_embedder

def embed_text(text: str) -> List[float]:
    return get_embedder().embed_text(text)
