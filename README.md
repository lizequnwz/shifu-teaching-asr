# Subtitle Studio · 开示字幕工作室

Upload a Mandarin recording, review passages against the original audio, and publish a Simplified Chinese TXT transcript and SRT captions. Gemini Transcribe provides word timestamps. OpenAI GPT Transcribe compares the wording and accepts confirmed terms such as names, Buddhist expressions, and scripture titles.

| Mode | Files and work | Speech API keys |
| --- | --- | --- |
| **Local** | The app runs on your computer. Its recordings, drafts, and published files stay in the local job directory. | Use your own keys. Audio clips are sent to Gemini and OpenAI. |
| **Online** | The app runs in a browser. Sign in to upload and review recordings in your private workspace. | The site operator supplies the keys and pays for calls within site and account limits. |

Both modes use the same transcription, review, and publication logic. They require no local speech-model downloads. This release processes recorded Mandarin audio and publishes the Chinese version.

Transcription uses Gemini and OpenAI APIs exclusively. There is no MLX, Qwen, Whisper, PyTorch, Hugging Face, or local forced-alignment runtime. FFmpeg handles audio decoding and clipping; WebRTC VAD checks speech coverage and helps choose clip boundaries. OpenCC normalizes Chinese text. These lightweight processing dependencies remain necessary for the API workflow.

## Run locally

Install FFmpeg, Python 3.12, Node.js 22 or newer, and uv. From the checkout root:

Use a fresh virtual environment when upgrading from the former local-model setup. Install only the requirements below; no model checkout or model cache is needed. Node.js is needed to build the frontend, not to run the built app.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-workflow.txt -r requirements-web.txt
npm --prefix web ci
npm --prefix web run build
```

Set `GEMINI_API_KEY` and `OPENAI_API_KEY` in your shell or in the project’s local `.env`. Keep that file private. The preflight checks the tools, imports, and key presence without sending audio.

```sh
.venv/bin/python scripts/subtitle_workflow.py preflight --cloud
.venv/bin/python scripts/subtitle_web.py
```

Open [the local workspace](http://127.0.0.1:8765/). It accepts recordings up to 120 minutes and 512 MiB. You can provide up to 100 confirmed terms, one per line. Local jobs live under `output/jobs/`. Keep the server running while processing. Restarting it resumes queued or interrupted jobs from checkpoints.

## Review and publish

Use the filters to find wording, speech coverage, timing, and cut-boundary issues. Accept or correct the wording, then replay and confirm each pending coverage or boundary issue. **Confirm after listening** records your decision for that issue. Use the whole-passage actions when the entire passage is inaudible or contains no speech.

The cue editor keeps timestamps for unchanged cues and pre-fills a time range for changed cues when it can. Review the cues marked **Check timing** and set their boundaries with the playhead. The exported SRT uses the media timeline, including silence before a video's audio track starts. Once the review is clear, use **Publish Chinese subtitles** to create the download revision. Further edits start a new draft.

For command-line and batch jobs:

```sh
.venv/bin/python scripts/subtitle_workflow.py run 'recordings/your-recording.m4a' --seconds 480 --terms 'glossary.txt'
.venv/bin/python scripts/subtitle_workflow.py resume 'output/jobs/JOB'
.venv/bin/python scripts/subtitle_workflow.py decide 'output/jobs/JOB' PASSAGE_ID suggestion
.venv/bin/python scripts/subtitle_workflow.py decide 'output/jobs/JOB' PASSAGE_ID --resolve-issue ISSUE_ID
.venv/bin/python scripts/subtitle_workflow.py publish 'output/jobs/JOB'
```

Omit `--seconds` for the full recording. A reviewable draft is a successful processing result. Check the returned `status` and `published_transcript` or `published_subtitles` fields when automating final delivery. If a provider charge is uncertain, the original reservation stays in the ledger. To deliberately resubmit, use the browser retry action or the exact recorded request ID with `resume JOB --retry-request ID`. The CLI accepts `--budget` for the local estimated allowance.

Read passage and issue IDs from the current draft's `review.json`. For batch ingestion, run `scripts/subtitle_workflow.py batch MANIFEST.jsonl`. Each JSON line contains an `audio` path and can include `terms`, `seconds`, `budget`, `core_seconds`, `context`, and `comparison`. Paths are relative to the manifest.

The default compares both providers on every window. Their calls can run together, and the completed reading is checkpointed if its peer pauses. Try `--window-seconds 90` to compare chunk lengths on the same audio. Use `--context '普通话佛学开示，引用金刚经。'` to test topic context for OpenAI along with your glossary. The experimental `--comparison targeted` checks windows with potential gaps, seam or timing issues, glossary hits, and about 10% of the remaining windows. Review the reported comparison window count and audio seconds along with accuracy when comparing runs.

## Deploy Online

The [Render Blueprint](render.yaml) combines a free web service with a paid background worker. The web service can have a cold start after inactivity. Supabase Free works for the pilot. Online recordings are limited to 50 MB (50,000,000 bytes) and 120 minutes. Budget separately for the worker, speech APIs, and any SMTP costs.

1. In Supabase, create a private Storage bucket named `recordings`. Allow 50 MB in both the project Storage settings and the bucket. Apply [the schema](migrations/001_online.sql) in the Supabase SQL Editor before deploying the web service. The worker also applies the schema in its predeploy step.
2. Enable email authentication and restrict registration to invited or pre-created confirmed accounts for the pilot. Supabase's default mailer sends a sign-in link to project team members. Configure custom SMTP before inviting users outside that team. If you want users to enter a code, add `{{ .Token }}` to the Magic Link or OTP template. Set the Site URL and allowed redirect to the final Studio HTTPS origin.
3. In Render, create the workspace environment group `subtitle-studio-online` and populate it with the shared values below. Create a Blueprint from this repository and set the speech keys on the worker. Confirm the actual web URL in `STUDIO_PUBLIC_ORIGIN`, then check `/health` and run a short authorized upload through review and download.

| Variable | Where to set it |
| --- | --- |
| `DATABASE_URL` | Shared group. Use the Supabase session pooler URI with `sslmode=require`. |
| `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`, `SUPABASE_SERVICE_ROLE_KEY` | Shared group. Use the Supabase project URL, publishable key, and server-only secret key, respectively. |
| `STUDIO_PUBLIC_ORIGIN` | The exact HTTPS site origin without a trailing path. |
| `ONLINE_ALLOWED_REGIONS` | Comma-separated ISO 3166-1 alpha-2 codes in regions supported by both speech providers. Account holders declare their region and confirm they are 18 or older. |
| `ONLINE_DAILY_BUDGET_USD` | The operator’s positive aggregate daily API budget. The budget day begins at 00:00 UTC. |
| `STUDIO_BUCKET` | The private bucket, defaulting to `recordings`. |
| `GEMINI_API_KEY`, `OPENAI_API_KEY` | Worker only. Use a paid Gemini project verified for this launch. |

Enter credentials in the Render and Supabase dashboards. Keep passwords, API keys, and local `.env` files out of Git. Each account can process 120 minutes in a rolling 30-day period. Jobs and exports expire 30 days after creation. Verify account isolation, email delivery, and resumable uploads before widening access. See [Supabase Storage limits](https://supabase.com/docs/guides/storage/uploads/file-limits), [Supabase SMTP](https://supabase.com/docs/guides/auth/auth-smtp), [Gemini regions](https://aistudio.google.com/docs/available-regions), and [OpenAI regions](https://developers.openai.com/api/docs/supported-countries).

For a pilot without hosting charges, create only a **Free Docker Web Service** in Render using this repository. Use `/health` for its health check and set the shared configuration on the service. Run `scripts/online_worker.py` with the local Python environment and the same settings on your Mac. The worker can read the speech keys from the local `.env`. Jobs stay queued when the Mac is offline. Speech API usage is billed separately.

Deployment to-do

- [ ] Move the transcription worker to an always-on remote host and verify a full transcription with the Mac off.

## Verify and measure release quality

```sh
uv pip install --python .venv/bin/python -r requirements-online.txt -r requirements-evaluation.txt
.venv/bin/python -m unittest discover -s tests -v
npm --prefix web run build
```

Set `STUDIO_TEST_POSTGRES_BIN` to a local PostgreSQL `bin` directory to also run the disposable database integration tests. They start their own cluster and mock Supabase and the speech providers.

Use authorized Mandarin recordings. Prepare at least 30 minutes of representative human-reviewed text and cue boundaries, then review one full recording longer than 70 minutes. Save the reviewed TXT and SRT for the same time scope and run:

```sh
.venv/bin/python scripts/evaluate_subtitles.py \
  --reference-txt gold.txt --reference-srt gold.srt \
  --candidate-txt zh.txt --candidate-srt zh.srt \
  --terms glossary.txt --check
```

The evaluator scores normalized Chinese character error rate and P95 start/end error at shared boundaries in the aligned reference and candidate text. It supports repeated cues in sequence and small groups of split or merged cues. The report includes coverage by cue count and duration, reference cue time without a candidate cue, and glossary precision and recall by timed occurrence. Listen to uncovered intervals to verify how much speech was missed. Pass `--audio-minutes` and `--review-minutes` to report active reviewer minutes per audio hour. Use `--text-only` with the two TXT files to compare raw drafts before timing. The clear-speech targets are at most 2% character error rate and at most 750 ms P95 cue-boundary error after review.
