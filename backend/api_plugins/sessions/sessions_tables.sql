-- Dialect MUST be sqlite, even if the database you use is different.
-- It is transpiled to the right dialect when executed.

CREATE TABLE IF NOT EXISTS "session" (
    "id" VARCHAR(255) PRIMARY KEY,
    "timestamp" DATETIME,
    "user_id" VARCHAR(255),
    FOREIGN KEY ("user_id") REFERENCES "users" ("email")
);
