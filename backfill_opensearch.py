"""
One-time backfill: read all rows already in Postgres and bulk-index them
directly into OpenSearch, using OpenSearch's _bulk API. This bypasses
Kafka/Debezium entirely -- appropriate here because we're not writing new
data to Postgres, so there's nothing for CDC to replay; we just need to
repopulate a freshly recreated OpenSearch index from data that's already
correct at the source of truth (Postgres).

Points at Aiven-hosted Postgres + OpenSearch (the "static clone" repo,
per project decision -- no live CDC pipeline deployed here).

Requires two environment variables (never hardcode credentials):
  PG_PASSWORD
  OPENSEARCH_PASSWORD

Run with:
  export PG_PASSWORD='...'
  export OPENSEARCH_PASSWORD='...'
  python3 backfill_opensearch.py
"""

import asyncio
import os
import ssl

import asyncpg
from opensearchpy import AsyncOpenSearch
from opensearchpy.helpers import async_bulk

PG_HOST = "search-postgres-searchproject.b.aivencloud.com"
PG_PORT = 25662
PG_USER = "avnadmin"
PG_DATABASE = "defaultdb"
PG_CA_CERT_PATH = "/Users/Nive/Desktop/postgres_ca.pem"

OPENSEARCH_HOST = "search-opensearch-searchproject.c.aivencloud.com"
OPENSEARCH_PORT = 25662
OPENSEARCH_USER = "avnadmin"
OPENSEARCH_INDEX = "bookdb.public.works"

BATCH_SIZE = 5000


async def row_generator(pool: asyncpg.Pool):
    """Stream rows from Postgres and yield them as OpenSearch bulk-index actions."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            # A named/server-side cursor -- avoids pulling the whole table
            # into memory at once for large row counts.
            cursor = conn.cursor(
                "SELECT id, title, subtitle, author_names, first_publish_year FROM works"
            )
            async for row in cursor:
                yield {
                    "_index": OPENSEARCH_INDEX,
                    "_id": row["id"],
                    "_source": {
                        "id": row["id"],
                        "title": row["title"],
                        "subtitle": row["subtitle"],
                        "author_names": list(row["author_names"] or []),
                        "first_publish_year": row["first_publish_year"],
                    },
                }


async def main():
    pg_password = os.environ["PG_PASSWORD"]
    opensearch_password = os.environ["OPENSEARCH_PASSWORD"]

    # Postgres: Aiven issues its own CA for TLS, so we point at the
    # downloaded ca.pem explicitly rather than relying on the system's
    # trusted CA store (which won't recognize it).
    pg_ssl_context = ssl.create_default_context(cafile=PG_CA_CERT_PATH)

    pg_pool = await asyncpg.create_pool(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=pg_password,
        database=PG_DATABASE,
        ssl=pg_ssl_context,
        min_size=1,
        max_size=2,
    )

    # OpenSearch: Aiven's *.aivencloud.com HTTPS endpoints use a publicly
    # trusted certificate (unlike Postgres's self-signed CA), so the
    # system's default trust store should verify it without needing a
    # custom CA file here. If this errors with a certificate verification
    # failure, that's the first thing to double check.
    os_client = AsyncOpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=(OPENSEARCH_USER, opensearch_password),
        use_ssl=True,
        verify_certs=True,
    )

    try:
        success, errors = await async_bulk(
            os_client,
            row_generator(pg_pool),
            chunk_size=BATCH_SIZE,
            raise_on_error=False,
        )
        print(f"Indexed {success:,} documents.")
        if errors:
            print(f"{len(errors):,} errors occurred. First few:")
            for err in errors[:5]:
                print(err)
    finally:
        await pg_pool.close()
        await os_client.close()


if __name__ == "__main__":
    asyncio.run(main())