ingest_rag_3:
	poetry run python -m backend.rag_3.ingest

serve:
	poetry run python -m app.server
