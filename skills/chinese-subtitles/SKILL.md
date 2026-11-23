---
name: chinese-subtitles
description: Create, resume, review, or publish Simplified Chinese transcripts and SRT subtitles for Mandarin recordings with Subtitle Studio.
---

# Chinese Subtitles

Use the existing `asr-translation` checkout. Start in the current folder if it contains `scripts/subtitle_workflow.py`, otherwise try `/Users/charlie/Repos/asr-translation`. The README describes local setup and Online deployment. Run terminal commands from the checkout root with `.venv/bin/python`.

## Start or resume a job

- Use `run AUDIO` for Mandarin audio. Add `--seconds 480` for an eight-minute pilot and `--terms FILE` for a UTF-8 glossary with one confirmed term per line.
- Use `batch MANIFEST.jsonl` for multiple recordings and `resume JOB` for an interrupted or paused job. Pilots and full recordings are separate jobs.
- Use `preflight --cloud` to check dependencies and credential availability without uploading audio. Keep credentials out of output.
- Both modes send audio clips to Gemini and OpenAI. State the selected scope and estimated allowance before running. Local defaults are $2 for a pilot and $10 for a full job. Retry an uncertain request only when authorized, using `resume JOB --retry-request ID`.

```sh
.venv/bin/python scripts/subtitle_workflow.py run 'recordings/example.m4a' --seconds 480
```

## Review and publish

Read `draft.json` under the job directory to find the current draft. Use its review list to identify timecoded clips, model candidates, and unresolved issues. Describe a clear queue as “no issues flagged.” Only describe audio as listened to when playback was heard.

Use the exact passage ID and the user's choice. `gemini`, `openai`, and `suggestion` choose a reading. `edited` accepts the complete replacement wording. Use `--text-stdin` for private or multiline text. Set manually timed cues in Studio or pass the JSON cue list through `--timings FILE`.

```sh
.venv/bin/python scripts/subtitle_workflow.py decide 'output/jobs/example' PASSAGE_ID suggestion
.venv/bin/python scripts/subtitle_workflow.py publish 'output/jobs/example'
```

Publishing makes a reviewed Chinese TXT and SRT revision available. Preserve earlier published revisions. Use Studio for listening, wording and timing edits, and downloads. Reuse the service if it is already running at `http://127.0.0.1:8765/`, or start it on loopback from the checkout.

```sh
.venv/bin/python scripts/subtitle_web.py
```
