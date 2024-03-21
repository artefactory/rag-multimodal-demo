-- Dialect MUST be sqlite, even if the database you use is different.
-- It is transpiled to the right dialect when executed.

CREATE TABLE IF NOT EXISTS "users" (
    "email" VARCHAR(255) PRIMARY KEY,
    "password" TEXT
);
