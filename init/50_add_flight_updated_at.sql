-- When each flight's scraped values last changed. New databases get the column from
-- 10_init_db.sql; `make migrate` adds it to older ones. Safe to run again.
ALTER TABLE arrivals ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE departures ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();
