CREATE TABLE prediction_audit (
    event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version VARCHAR(64) NOT NULL,
    prediction_source VARCHAR(32) NOT NULL
        CHECK (prediction_source IN ('single', 'batch')),
    total_count INTEGER NOT NULL CHECK (total_count > 0),
    low_count INTEGER NOT NULL CHECK (low_count >= 0),
    medium_count INTEGER NOT NULL CHECK (medium_count >= 0),
    high_count INTEGER NOT NULL CHECK (high_count >= 0),
    CHECK (total_count = low_count + medium_count + high_count)
);

CREATE INDEX prediction_audit_created_at_idx
    ON prediction_audit (created_at DESC);

-- Replace carepredict_app with the database role used by the local app.
GRANT SELECT, INSERT, DELETE ON prediction_audit TO carepredict_app;
GRANT USAGE, SELECT ON SEQUENCE prediction_audit_event_id_seq TO carepredict_app;
