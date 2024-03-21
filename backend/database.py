from logging import Logger
from pathlib import Path
from typing import Any

import sqlglot
from dbutils.pooled_db import PooledDB
from sqlalchemy.engine.url import make_url

from backend import DATABASE_URL
import logging

POOL = None


def get_logger() -> Logger:
    return logging.getLogger(__name__)


class Database:
    """Handles database operations.

    It manages the connection pool, executes queries, and handles transactions.
    The class uses a context manager to ensure that database connections are properly
    opened and closed. It supports SQLite, PostgreSQL, and MySQL databases.

    Attributes:
        connection_string (str): The database connection string.
        logger (Logger): The logger instance for logging messages.
        url (URL): The parsed URL object of the connection string.
        pool (PooledDB): The connection pool for database connections.
        conn (Connection): The current database connection.
        DIALECT_PLACEHOLDERS (dict): Mapping of database dialects to their placeholder symbols.
    """

    DIALECT_PLACEHOLDERS = {
        "sqlite": "?",
        "postgresql": "%s",
        "mysql": "%s",
    }

    def __init__(self, connection_string: str = None, logger: Logger = None):
        self.connection_string = connection_string or DATABASE_URL
        self.logger = logger or get_logger()

        self.url = make_url(self.connection_string)

        self.logger.debug("Creating connection pool")
        global POOL
        POOL = POOL or self._create_pool()  # Makes the pool a singleton
        self.pool = POOL
        self.conn = None

    def __enter__(self) -> "Database":
        self.logger.debug("Getting connection from pool")
        self.conn = self.pool.connection()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> None:
        if self.conn:
            if exc_type:
                self.logger.error(
                    "Transaction failed", exc_info=(exc_type, exc_value, traceback)
                )
                self.conn.rollback()
            else:
                self.conn.commit()
            self.logger.debug("Returning connection to pool")
            self.conn.close()
            self.conn = None

    def execute(self, query: str, params: tuple | None = None) -> Any:
        cursor = self.conn.cursor()
        try:
            query = query.replace("?", self.DIALECT_PLACEHOLDERS[self.url.drivername])
            self.logger.debug(f"Executing query: {query}")
            cursor.execute(query, params or ())
            return cursor
        except Exception as e:
            cursor.close()
            self.logger.exception("Query execution failed", exc_info=e)
            raise

    def fetchone(self, query: str, params: tuple | None = None) -> tuple | None:
        cursor = self.execute(query, params)
        try:
            return cursor.fetchone()
        finally:
            cursor.close()

    def fetchall(self, query: str, params: tuple | None = None) -> list:
        cursor = self.execute(query, params)
        try:
            return cursor.fetchall()
        finally:
            cursor.close()

    def initialize_schema(self):
        try:
            self.logger.debug("Initializing database schema")
            sql_script = Path(__file__).parent.joinpath("db_init.sql").read_text()
            transpiled_sql = sqlglot.transpile(
                sql_script,
                read="sqlite",
                write=self.url.drivername.replace("postgresql", "postgres"),
            )
            for statement in transpiled_sql:
                self.execute(statement)
            self.logger.info(
                f"Database schema initialized successfully for {self.url.drivername}"
            )
        except Exception as e:
            self.logger.exception("Schema initialization failed", exc_info=e)
            raise

    def run_script(self, path: Path):
        try:
            self.logger.debug(f"Running Database script at {str(path)}")
            sql_script = path.read_text()
            transpiled_sql = sqlglot.transpile(
                sql_script,
                read="sqlite",
                write=self.url.drivername.replace("postgresql", "postgres"),
            )
            for statement in transpiled_sql:
                self.execute(statement)
            self.logger.info(
                f"Successfuly ran script at {path} for {self.url.drivername}"
            )
        except Exception as e:
            self.logger.exception(
                f"Failed to execute the script {path} for {self.url.drivername}",
                exc_info=e,
            )
            raise

    def _create_pool(self) -> PooledDB:
        if self.connection_string.startswith("sqlite:///"):
            import sqlite3

            Path(self.connection_string.replace("sqlite:///", "")).parent.mkdir(
                parents=True, exist_ok=True
            )
            return PooledDB(
                creator=sqlite3,
                database=self.connection_string.replace("sqlite:///", ""),
                maxconnections=5,
            )
        elif self.connection_string.startswith("postgresql://"):
            import psycopg2

            return PooledDB(
                creator=psycopg2, dsn=self.connection_string, maxconnections=5
            )
        elif self.connection_string.startswith(
            "mysql://"
        ) or self.connection_string.startswith("mysql+pymysql://"):
            import mysql.connector

            return PooledDB(
                creator=mysql.connector,
                user=self.url.username,
                password=self.url.password,
                host=self.url.host,
                port=self.url.port,
                database=self.url.database,
                maxconnections=5,
            )
        elif self.connection_string.startswith("sqlserver://"):
            import pyodbc

            return PooledDB(
                creator=pyodbc,
                dsn=self.connection_string.replace("sqlserver://", ""),
                maxconnections=5,
            )
        else:
            raise ValueError(f"Unsupported database type: {self.url.drivername}")
