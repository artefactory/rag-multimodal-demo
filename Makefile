ingest_rag_1:
	poetry run python -m backend.rag_1.ingest

ingest_rag_2:
	poetry run python -m backend.rag_2.ingest

ingest_rag_3:
	poetry run python -m backend.rag_3.ingest

serve:
	poetry run python -m app.server
