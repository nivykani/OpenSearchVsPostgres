"""
Search comparison API.

Two independent endpoints -- /search/postgres and /search/opensearch --
rather than one combined endpoint. This lets the frontend fire both
requests at once and render whichever finishes first immediately, instead
of waiting for the slower of the two before showing anything.

Run locally with:
  uvicorn main:app --reload --port 8000
"""

import json
import os
import ssl
import time
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from opensearchpy import AsyncOpenSearch
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# --- Aiven connection details ---
PG_HOST = "search-postgres-searchproject.b.aivencloud.com"
PG_PORT = 25662
PG_USER = "avnadmin"
PG_DATABASE = "defaultdb"
PG_CA_CERT_PATH = os.environ.get("PG_CA_CERT_PATH", "/etc/secrets/postgres_ca.pem")

OPENSEARCH_HOST = "search-opensearch-searchproject.c.aivencloud.com"
OPENSEARCH_PORT = 25662
OPENSEARCH_USER = "avnadmin"
OPENSEARCH_INDEX = "bookdb.public.works"

RESULT_LIMIT = 20


class WorkResult(BaseModel):
    id: str
    title: str
    subtitle: str | None
    author_names: list[str]
    first_publish_year: int | None


class OpenSearchResponse(BaseModel):
    results: list[WorkResult]
    time_ms: float


class PostgresResponse(BaseModel):
    results: list[WorkResult]
    time_ms: float
    explain: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Created once at startup, reused for every request -- this is the
    # connection pooling you asked about. app.state is FastAPI's sanctioned
    # place to stash long-lived objects that request handlers need to reach.
    #
    # Credentials come from environment variables, never hardcoded --
    # set PG_PASSWORD and OPENSEARCH_PASSWORD before starting the app.
    pg_ssl_context = ssl.create_default_context(cafile=PG_CA_CERT_PATH)

    app.state.pg_pool = await asyncpg.create_pool(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=os.environ["PG_PASSWORD"],
        database=PG_DATABASE,
        ssl=pg_ssl_context,
        min_size=2,
        max_size=10,
    )
    app.state.os_client = AsyncOpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=(OPENSEARCH_USER, os.environ["OPENSEARCH_PASSWORD"]),
        use_ssl=True,
        verify_certs=True,
    )
    yield
    await app.state.pg_pool.close()
    await app.state.os_client.close()


app = FastAPI(lifespan=lifespan)

# Rate limiting: keyed by client IP, applied per endpoint below. This
# matters more than it might for a typical endpoint, because a slow
# unindexed Postgres scan (measured at ~23s in testing) can hold a pooled
# connection open for a long time -- a handful of rapid repeat requests
# from one client could exhaust the whole pool (max_size=10) and degrade
# the app for every other visitor. Each endpoint now gets its own limit
# bucket, since a client could otherwise hammer just the Postgres route
# while staying under a shared limit.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Locked to the actual production frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nivykani.com", "https://www.nivykani.com"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def find_scan_node(plan_node: dict) -> dict:
    """
    EXPLAIN plans are trees -- the top node is often a wrapper (Limit,
    Aggregate, etc.), not the actual scan. Walk down through child "Plans"
    until we hit something that looks like a real scan node, so we report
    what Postgres is actually doing to find rows, not just the outermost
    wrapper.
    """
    node_type = plan_node.get("Node Type", "")
    if "Scan" in node_type:
        return plan_node
    for child in plan_node.get("Plans", []):
        found = find_scan_node(child)
        if found:
            return found
    return plan_node  # fall back to whatever we started with


@app.get("/search/opensearch", response_model=OpenSearchResponse)
@limiter.limit("20/minute")
async def search_opensearch(request: Request, q: str = Query(..., min_length=1)):
    body = {
        "query": {
            "multi_match": {
                "query": q,
                "fields": ["title", "subtitle", "author_names"],
            }
        },
        "size": RESULT_LIMIT,
    }

    start = time.perf_counter()
    response = await app.state.os_client.search(index=OPENSEARCH_INDEX, body=body)
    elapsed_ms = (time.perf_counter() - start) * 1000

    results = [
        WorkResult(
            id=hit["_source"]["id"],
            title=hit["_source"]["title"],
            subtitle=hit["_source"].get("subtitle"),
            author_names=hit["_source"].get("author_names") or [],
            first_publish_year=hit["_source"].get("first_publish_year"),
        )
        for hit in response["hits"]["hits"]
    ]

    return OpenSearchResponse(results=results, time_ms=elapsed_ms)


@app.get("/search/postgres", response_model=PostgresResponse)
@limiter.limit("20/minute")
async def search_postgres(request: Request, q: str = Query(..., min_length=1)):
    sql = """
        SELECT id, title, subtitle, author_names, first_publish_year
        FROM works
        WHERE title ILIKE '%' || $1 || '%'
           OR subtitle ILIKE '%' || $1 || '%'
           OR EXISTS (
               SELECT 1 FROM unnest(author_names) AS a
               WHERE a ILIKE '%' || $1 || '%'
           )
        LIMIT $2
    """
    explain_sql = "EXPLAIN (ANALYZE, FORMAT JSON)\n" + sql

    async with app.state.pg_pool.acquire() as conn:
        rows = await conn.fetch(sql, q, RESULT_LIMIT)
        explain_row = await conn.fetchval(explain_sql, q, RESULT_LIMIT)

    results = [
        WorkResult(
            id=row["id"],
            title=row["title"],
            subtitle=row["subtitle"],
            author_names=list(row["author_names"] or []),
            first_publish_year=row["first_publish_year"],
        )
        for row in rows
    ]

    # asyncpg returns the EXPLAIN JSON output as a JSON-encoded string;
    # it's a list containing one plan object.
    plan = json.loads(explain_row)[0]
    scan_node = find_scan_node(plan["Plan"])
    explain = {
        "top_node_type": plan["Plan"].get("Node Type"),
        "scan_node_type": scan_node.get("Node Type"),
        "execution_time_ms": plan.get("Execution Time"),
        "planning_time_ms": plan.get("Planning Time"),
        "actual_rows": plan["Plan"].get("Actual Rows"),
    }

    return PostgresResponse(
        results=results,
        time_ms=explain["execution_time_ms"],
        explain=explain,
    )