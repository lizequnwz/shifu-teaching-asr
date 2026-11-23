CREATE SCHEMA IF NOT EXISTS studio;
REVOKE ALL ON SCHEMA studio FROM PUBLIC;

CREATE TABLE IF NOT EXISTS studio.jobs (
    id uuid PRIMARY KEY,
    owner_id uuid NOT NULL,
    name text NOT NULL,
    source_bytes bigint NOT NULL CHECK (source_bytes BETWEEN 1 AND 536870912),
    mime_type text NOT NULL,
    terms jsonb NOT NULL,
    status text NOT NULL CHECK (status IN (
        'uploading', 'queued', 'processing', 'needs_review', 'ready_to_publish',
        'no_speech', 'published', 'paused', 'failed', 'deleted'
    )),
    stage text NOT NULL,
    completed integer,
    total integer,
    error text,
    retry_request text,
    authorized_retry text,
    rate_limit_count integer NOT NULL DEFAULT 0,
    available_at timestamptz NOT NULL DEFAULT now(),
    lease uuid,
    lease_until timestamptz,
    duration_seconds integer CHECK (duration_seconds BETWEEN 1 AND 7200),
    duration_reserved_at timestamptz,
    manifest jsonb,
    data jsonb,
    summary jsonb,
    passages jsonb,
    clip_prefix text,
    decisions jsonb NOT NULL DEFAULT '{}'::jsonb,
    revision uuid NOT NULL,
    published_revision uuid,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL DEFAULT (now() + interval '30 days'),
    cleanup_at timestamptz,
    upload_grant_until timestamptz
);

CREATE INDEX IF NOT EXISTS studio_jobs_owner ON studio.jobs (owner_id, created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS studio_ready_jobs ON studio.jobs (available_at, id)
    WHERE status IN ('queued', 'processing');
CREATE UNIQUE INDEX IF NOT EXISTS studio_one_active_job ON studio.jobs (owner_id)
    WHERE status IN ('uploading', 'queued', 'processing');

CREATE TABLE IF NOT EXISTS studio.windows (
    job_id uuid REFERENCES studio.jobs (id) ON DELETE CASCADE,
    name text NOT NULL,
    result jsonb NOT NULL,
    PRIMARY KEY (job_id, name)
);

CREATE TABLE IF NOT EXISTS studio.publications (
    job_id uuid REFERENCES studio.jobs (id) ON DELETE CASCADE,
    revision uuid NOT NULL,
    object_prefix text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (job_id, revision)
);

CREATE TABLE IF NOT EXISTS studio.requests (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id uuid REFERENCES studio.jobs (id) ON DELETE SET NULL,
    fingerprint text,
    provider text NOT NULL,
    model text NOT NULL,
    purpose text NOT NULL,
    request jsonb,
    state text NOT NULL CHECK (state IN ('pending', 'completed', 'invalid_response', 'rejected', 'rate_limited')),
    reserved_usd double precision NOT NULL CHECK (reserved_usd >= 0),
    estimated_usd double precision,
    raw jsonb,
    result jsonb,
    usage jsonb,
    error_kind text,
    provider_request_id text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS studio_request_fingerprint ON studio.requests (job_id, fingerprint, id DESC);
CREATE INDEX IF NOT EXISTS studio_request_day ON studio.requests (created_at);
