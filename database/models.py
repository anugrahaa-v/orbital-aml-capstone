"""SQLAlchemy Core schema (SQLite for local dev, PostgreSQL in production).
The PostgreSQL-native DDL with BRIN indexes lives in database/schema.sql."""
from sqlalchemy import (MetaData, Table, Column, Integer, BigInteger, Float, String, Text, DateTime,
                        Boolean, JSON, ForeignKey, Index, UniqueConstraint)

metadata = MetaData()

satellite_objects = Table(
    "satellite_objects", metadata,
    Column("norad_id", BigInteger, primary_key=True, autoincrement=False),
    Column("name", String(128), nullable=False),
    Column("object_id", String(32)),
    Column("source_group", String(64), nullable=False),
    Column("obj_class", String(16), nullable=False),
    Column("class_source", String(32), nullable=False),
    Column("first_seen_utc", DateTime, nullable=False),
    Column("last_seen_utc", DateTime, nullable=False),
)

orbital_element_history = Table(
    "orbital_element_history", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("norad_id", BigInteger, ForeignKey("satellite_objects.norad_id"), nullable=False),
    Column("epoch_utc", DateTime, nullable=False),
    Column("retrieved_utc", DateTime, nullable=False),
    Column("dataset_version", String(64), nullable=False),
    Column("mean_motion", Float, nullable=False), Column("eccentricity", Float, nullable=False),
    Column("inclination", Float, nullable=False), Column("raan", Float, nullable=False),
    Column("arg_perigee", Float, nullable=False), Column("mean_anomaly", Float, nullable=False),
    Column("bstar", Float, nullable=False), Column("mean_motion_dot", Float),
    Column("mean_motion_ddot", Float),
    Column("raw_omm", JSON, nullable=False),
    UniqueConstraint("norad_id", "epoch_utc", name="uq_history_norad_epoch"),
    Index("ix_history_norad_epoch", "norad_id", "epoch_utc"),
    Index("ix_history_epoch", "epoch_utc"),
)

propagated_positions = Table(
    "propagated_positions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("norad_id", BigInteger, nullable=False),
    Column("time_utc", DateTime, nullable=False),
    Column("frame", String(8), nullable=False, default="TEME"),
    Column("x_km", Float, nullable=False), Column("y_km", Float, nullable=False), Column("z_km", Float, nullable=False),
    Column("vx_kms", Float), Column("vy_kms", Float), Column("vz_kms", Float),
    Column("source_epoch_utc", DateTime, nullable=False),
    Index("ix_prop_norad_time", "norad_id", "time_utc"),
)

conjunction_screening = Table(
    "conjunction_screening", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_utc", DateTime, nullable=False),
    Column("primary_id", BigInteger, nullable=False), Column("secondary_id", BigInteger, nullable=False),
    Column("tca_utc", DateTime, nullable=False),
    Column("miss_distance_km", Float, nullable=False), Column("relative_speed_kms", Float, nullable=False),
    Column("threshold_km", Float, nullable=False), Column("category", String(16), nullable=False),
    Column("pc_available", Boolean, nullable=False, default=False), Column("pc", Float),
    Column("config", JSON),
    Index("ix_conj_primary_tca", "primary_id", "tca_utc"),
    Index("ix_conj_miss", "miss_distance_km"),
)

model_experiments = Table(
    "model_experiments", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(40), nullable=False, unique=True),
    Column("created_utc", DateTime, nullable=False),
    Column("kind", String(32), nullable=False),          # regression|classification|clustering|deep_learning|...
    Column("model_name", String(64)),
    Column("target", String(64)),
    Column("seed", Integer), Column("params", JSON), Column("metrics", JSON),
    Column("provenance", JSON), Column("data_label", String(32)),   # observed|derived|simulated
    Index("ix_exp_kind_created", "kind", "created_utc"),
)

predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(40), nullable=False),
    Column("norad_id", BigInteger), Column("time_utc", DateTime),
    Column("target", String(64), nullable=False),
    Column("y_true", Float), Column("y_pred", Float, nullable=False),
    Index("ix_pred_run", "run_id"),
)

rl_experiments = Table(
    "rl_experiments", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(40), nullable=False),
    Column("created_utc", DateTime, nullable=False),
    Column("policy", String(32), nullable=False), Column("seed", Integer, nullable=False),
    Column("final_separation_km", Float), Column("fuel_used_ms", Float), Column("total_reward", Float),
    Column("config", JSON), Column("trajectory", JSON),
    Index("ix_rl_run", "run_id"),
)

source_registry = Table(
    "source_registry", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source", String(64), nullable=False), Column("query_url", Text, nullable=False),
    Column("retrieved_utc", String(24), nullable=False), Column("dataset_version", String(64), nullable=False),
    Column("n_records_raw", Integer, nullable=False), Column("access_conditions", Text),
    Column("attribution", Text), Column("from_cache", Boolean),
)
