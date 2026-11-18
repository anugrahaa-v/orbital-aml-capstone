-- ORBITALX AI: PostgreSQL schema (same logical model as database/models.py).
-- Time-series design: orbital_element_history is append-only, keyed (norad_id, epoch_utc);
-- BRIN index on epoch for cheap range scans, btree on (norad_id, epoch_utc) for per-object series.
-- For very large histories, partition by RANGE (epoch_utc) monthly or use TimescaleDB create_hypertable().

CREATE TABLE IF NOT EXISTS satellite_objects (
  norad_id BIGINT PRIMARY KEY,
  name VARCHAR(128) NOT NULL,
  object_id VARCHAR(32),
  source_group VARCHAR(64) NOT NULL,
  obj_class VARCHAR(16) NOT NULL CHECK (obj_class IN ('PAYLOAD','DEBRIS','ROCKET_BODY')),
  class_source VARCHAR(32) NOT NULL,
  first_seen_utc TIMESTAMP NOT NULL,
  last_seen_utc TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS orbital_element_history (
  id BIGSERIAL PRIMARY KEY,
  norad_id BIGINT NOT NULL REFERENCES satellite_objects(norad_id),
  epoch_utc TIMESTAMP NOT NULL,
  retrieved_utc TIMESTAMP NOT NULL,
  dataset_version CHAR(64) NOT NULL,
  mean_motion DOUBLE PRECISION NOT NULL, eccentricity DOUBLE PRECISION NOT NULL,
  inclination DOUBLE PRECISION NOT NULL, raan DOUBLE PRECISION NOT NULL,
  arg_perigee DOUBLE PRECISION NOT NULL, mean_anomaly DOUBLE PRECISION NOT NULL,
  bstar DOUBLE PRECISION NOT NULL, mean_motion_dot DOUBLE PRECISION, mean_motion_ddot DOUBLE PRECISION,
  raw_omm JSONB NOT NULL,
  CONSTRAINT uq_history_norad_epoch UNIQUE (norad_id, epoch_utc)
);
CREATE INDEX IF NOT EXISTS ix_history_epoch_brin ON orbital_element_history USING BRIN (epoch_utc);

CREATE TABLE IF NOT EXISTS propagated_positions (
  id BIGSERIAL PRIMARY KEY, norad_id BIGINT NOT NULL, time_utc TIMESTAMP NOT NULL,
  frame VARCHAR(8) NOT NULL DEFAULT 'TEME',
  x_km DOUBLE PRECISION NOT NULL, y_km DOUBLE PRECISION NOT NULL, z_km DOUBLE PRECISION NOT NULL,
  vx_kms DOUBLE PRECISION, vy_kms DOUBLE PRECISION, vz_kms DOUBLE PRECISION,
  source_epoch_utc TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_prop_norad_time ON propagated_positions (norad_id, time_utc);

CREATE TABLE IF NOT EXISTS conjunction_screening (
  id BIGSERIAL PRIMARY KEY, run_utc TIMESTAMP NOT NULL,
  primary_id BIGINT NOT NULL, secondary_id BIGINT NOT NULL, tca_utc TIMESTAMP NOT NULL,
  miss_distance_km DOUBLE PRECISION NOT NULL, relative_speed_kms DOUBLE PRECISION NOT NULL,
  threshold_km DOUBLE PRECISION NOT NULL, category VARCHAR(16) NOT NULL,
  pc_available BOOLEAN NOT NULL DEFAULT FALSE, pc DOUBLE PRECISION, config JSONB
);
CREATE INDEX IF NOT EXISTS ix_conj_primary_tca ON conjunction_screening (primary_id, tca_utc);
CREATE INDEX IF NOT EXISTS ix_conj_miss ON conjunction_screening (miss_distance_km);

CREATE TABLE IF NOT EXISTS model_experiments (
  id BIGSERIAL PRIMARY KEY, run_id VARCHAR(40) NOT NULL UNIQUE, created_utc TIMESTAMP NOT NULL,
  kind VARCHAR(32) NOT NULL, model_name VARCHAR(64), target VARCHAR(64), seed INTEGER,
  params JSONB, metrics JSONB, provenance JSONB, data_label VARCHAR(32)
);
CREATE INDEX IF NOT EXISTS ix_exp_kind_created ON model_experiments (kind, created_utc DESC);

CREATE TABLE IF NOT EXISTS predictions (
  id BIGSERIAL PRIMARY KEY, run_id VARCHAR(40) NOT NULL, norad_id BIGINT, time_utc TIMESTAMP,
  target VARCHAR(64) NOT NULL, y_true DOUBLE PRECISION, y_pred DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_pred_run ON predictions (run_id);

CREATE TABLE IF NOT EXISTS rl_experiments (
  id BIGSERIAL PRIMARY KEY, run_id VARCHAR(40) NOT NULL, created_utc TIMESTAMP NOT NULL,
  policy VARCHAR(32) NOT NULL, seed INTEGER NOT NULL, final_separation_km DOUBLE PRECISION,
  fuel_used_ms DOUBLE PRECISION, total_reward DOUBLE PRECISION, config JSONB, trajectory JSONB
);
CREATE INDEX IF NOT EXISTS ix_rl_run ON rl_experiments (run_id);

CREATE TABLE IF NOT EXISTS source_registry (
  id BIGSERIAL PRIMARY KEY, source VARCHAR(64) NOT NULL, query_url TEXT NOT NULL,
  retrieved_utc VARCHAR(24) NOT NULL, dataset_version CHAR(64) NOT NULL, n_records_raw INTEGER NOT NULL,
  access_conditions TEXT, attribution TEXT, from_cache BOOLEAN
);
