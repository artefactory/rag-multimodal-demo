ingest_rag_3:
	poetry run python -m backend.rag_3.ingest

launch_app:
	poetry run python -m app.server
