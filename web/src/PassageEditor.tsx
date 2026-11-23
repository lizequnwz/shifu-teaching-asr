import { useCallback, useEffect, useRef, useState, type Ref } from 'react'
import { Check, ChevronRight, Clock3, Play, Plus } from 'lucide-react'
import Player, { clock, type PlaybackRange, type ReplayRequest } from './Player'
import {
  ApiError, api, sources, type Choice, type Issue, type Job, type Passage,
  type PlayableClip, type SavedReview, type Timing, type TimingTemplate,
} from './api'
import { issueDescription, type Translate } from './i18n'

const exactMs = (seconds: number) => Number((seconds * 1000).toFixed(7))
const sampleRate = 16_000
type CueInput = { id: number; text: string; start: string; end: string; needsConfirmation?: boolean }

function issueRange(issue: Issue, clip: PlaybackRange): PlaybackRange | undefined {
  if (issue.start_sample === undefined || issue.end_sample === undefined) return
  const start = issue.start_sample / sampleRate
  const end = issue.end_sample / sampleRate
  if (!(clip.start_seconds <= start && start <= end && end <= clip.end_seconds)) return
  return {
    start_seconds: Math.max(clip.start_seconds, start - 0.75),
    end_seconds: Math.min(clip.end_seconds, end + 0.75),
  }
}

function needsReadingTime(text: string, seconds: number) {
  const characters = Array.from(text).reduce((count, char) =>
    count + (/\s/u.test(char) ? 0 : char.codePointAt(0)! < 128 ? 0.5 : 1), 0)
  return seconds > 0 && characters / seconds > 9
}

function highlighted(text: string, spans: { start: number; end: number }[]) {
  const chars = Array.from(text)
  const pieces = []
  let end = 0
  for (const span of [...spans].sort((a, b) => a.start - b.start)) {
    const start = Math.max(end, span.start)
    if (span.end <= start) continue
    if (start > end) pieces.push(chars.slice(end, start).join(''))
    pieces.push(<mark key={start}>{chars.slice(start, span.end).join('')}</mark>)
    end = span.end
  }
  pieces.push(chars.slice(end).join(''))
  return pieces
}

export default function PassageEditor({ job, item, compact, speed, onSpeedChange, heading, t, busy, onBusy, onSave, onReload, onDirty }: {
  job: Job
  item: Passage
  compact: boolean
  speed: number
  onSpeedChange: (value: number) => void
  heading: Ref<HTMLHeadingElement>
  t: Translate
  busy: boolean
  onBusy: (value: boolean) => void
  onSave: (result: SavedReview, next: boolean) => Promise<void>
  onReload: () => Promise<void>
  onDirty: (value: boolean) => void
}) {
  const [wording, setWording] = useState(item.selected_text)
  const [choice, setChoice] = useState<Choice>('suggestion')
  const [playable, setPlayable] = useState<PlayableClip | null>(null)
  const [ready, setReady] = useState(false)
  const [clipError, setClipError] = useState('')
  const [clipAttempt, setClipAttempt] = useState(0)
  const [replay, setReplay] = useState<ReplayRequest>()
  const [timing, setTiming] = useState<CueInput[] | null>(null)
  const [stale, setStale] = useState(false)
  const [error, setError] = useState('')
  const playhead = useRef<number | null>(null)
  const nextCue = useRef(0)
  const replayId = useRef(0)
  const priorTimings = useRef(item.timings ?? [])
  const clip = item.clips[0]
  const clipId = clip?.id
  const clipStart = clip?.start_seconds
  const clipEnd = clip?.end_seconds
  const clipUrl = clip?.url
  const dirty = choice !== 'suggestion' || wording !== item.selected_text || timing !== null
  const saveDisabled = busy || stale || (choice === 'edited' && !wording.trim() && !timing)
  useEffect(() => {
    onDirty(dirty)
    return () => onDirty(false)
  }, [dirty, onDirty])
  useEffect(() => {
    let active = true
    setPlayable(null)
    setReady(false)
    playhead.current = null
    setClipError('')
    if (clipId && clipStart !== undefined && clipEnd !== undefined) void api.clip(job.id, {
      id: clipId, start_seconds: clipStart, end_seconds: clipEnd, url: clipUrl,
    }).then(
      (resolved) => { if (active) setPlayable(resolved) },
      () => { if (active) setClipError(t('clipFailed')) },
    )
    return () => { active = false }
  }, [job.id, clipId, clipStart, clipEnd, clipUrl, clipAttempt, t])
  const reportPlayhead = useCallback((_url: string, seconds: number | null) => {
    const wasReady = playhead.current !== null
    playhead.current = seconds
    if (wasReady !== (seconds !== null)) setReady(seconds !== null)
  }, [])

  function choose(next: Choice, text: string) {
    setChoice(next)
    setWording(text)
    setTiming(null)
    setError('')
  }
  function showTiming(text: string, cues: Timing[] | TimingTemplate[]) {
    setTiming(cues.length ? cues.map((cue) => ({
      id: nextCue.current++, text: cue.text,
      start: cue.start_ms === null ? '' : (cue.start_ms / 1000).toFixed(3),
      end: cue.end_ms === null ? '' : (cue.end_ms / 1000).toFixed(3),
      needsConfirmation: 'needs_confirmation' in cue && cue.needs_confirmation,
    })) : [{ id: nextCue.current++, text, start: '', end: '' }])
  }
  function openTiming() {
    const saved = item.timings?.length ? item.timings : priorTimings.current
    if (wording !== item.selected_text && (saved.length > 1 || item.timing_template?.length)) {
      void save(false, true)
      return
    }
    if (wording === item.selected_text) {
      showTiming(wording, item.timing_template?.length ? item.timing_template : item.timings ?? [])
    } else {
      showTiming(wording, saved.length === 1 ? [{ ...saved[0], text: wording }] : [])
    }
  }
  function replayRange(range: PlaybackRange) {
    setReplay({ ...range, id: ++replayId.current })
  }

  function confirmWholePassage(next: 'inaudible' | 'no_speech', text: string) {
    if (window.confirm(t('wholePassageConfirm'))) choose(next, text)
  }
  function updateCue(id: number, key: 'text' | 'start' | 'end', value: string) {
    setError('')
    setTiming((current) => current?.map((cue) => cue.id === id ? { ...cue, [key]: value } : cue) ?? null)
  }
  function usePlayhead(id: number, key: 'start' | 'end') {
    if (playhead.current === null) return
    const ms = Math.round(playhead.current * 1000)
    if (ms < Math.ceil(exactMs(item.start_seconds)) || ms > Math.floor(exactMs(item.end_seconds))) {
      setError(t('playheadOutside'))
      return
    }
    updateCue(id, key, (ms / 1000).toFixed(3))
  }

  function splitCue(index: number, caret: number) {
    if (!timing || playhead.current === null) return
    const cue = timing[index]
    const split = (Math.round(playhead.current * 1000) / 1000).toFixed(3)
    const left = cue.text.slice(0, caret).trimEnd()
    const right = cue.text.slice(caret).trimStart()
    const inWord = /[a-z0-9]$/i.test(cue.text.slice(0, caret)) && /^[a-z0-9]/i.test(cue.text.slice(caret))
    if (!left.trim() || !right.trim() || inWord || !cue.start || !cue.end ||
        !(Number(cue.start) < Number(split) && Number(split) < Number(cue.end))) {
      setError(t('timingSplitHint'))
      return
    }
    setError('')
    setTiming([...timing.slice(0, index), { ...cue, text: left, end: split },
      { id: nextCue.current++, text: right, start: split, end: cue.end, needsConfirmation: cue.needsConfirmation },
      ...timing.slice(index + 1)])
  }

  function mergeCue(index: number) {
    if (!timing || !timing[index + 1]) return
    const [left, right] = timing.slice(index, index + 2)
    setError('')
    setTiming([...timing.slice(0, index), { ...left, text: `${left.text}\n${right.text}`, end: right.end,
      needsConfirmation: left.needsConfirmation || right.needsConfirmation },
      ...timing.slice(index + 2)])
  }

  function saveError(problem: unknown) {
    if (problem instanceof ApiError && problem.status === 409) {
      setStale(true)
      setError(t('stale'))
    } else {
      setError(problem instanceof Error ? problem.message : String(problem))
    }
  }

  async function confirmIssue(issue: Issue) {
    if (!job.revision || busy || stale) return
    setError('')
    onBusy(true)
    try {
      const updated = await api.save(job.id, item.id, { resolved_issues: [issue.id], revision: job.revision })
      await onSave(updated, false)
    } catch (problem) {
      saveError(problem)
    } finally {
      onBusy(false)
    }
  }

  async function save(next: boolean, repairAfterSave = false) {
    if (!job.revision || saveDisabled) return
    if (!dirty && item.human_reviewed) {
      onBusy(true)
      try {
        await onSave({ job, passage: item }, next)
      } finally {
        onBusy(false)
      }
      return
    }
    let decision = choice
    let text: string | undefined
    let timings: Timing[] | undefined
    setError('')
    if (timing) {
      let previous = Math.ceil(exactMs(item.start_seconds))
      timings = []
      for (const cue of timing) {
        const start_ms = Math.round(Number(cue.start) * 1000)
        const end_ms = Math.round(Number(cue.end) * 1000)
        if (!cue.start.trim() || !cue.end.trim() || !cue.text.trim() ||
            !Number.isFinite(start_ms) || !Number.isFinite(end_ms) ||
            start_ms < previous || end_ms <= start_ms || end_ms > Math.floor(exactMs(item.end_seconds))) {
          setError(t('timingInvalid'))
          return
        }
        timings.push({ text: cue.text.trim(), start_ms, end_ms })
        previous = end_ms
      }
      decision = 'edited'
      text = timings.map((cue) => cue.text).join('\n')
    } else if (decision === 'edited' ||
        (decision === 'suggestion' && wording !== item.suggested_text && wording.trim())) {
      decision = 'edited'
      text = wording.trim()
    } else if (decision === 'suggestion' && !wording.trim()) {
      setError(t('noSpeechRequired'))
      return
    }
    onBusy(true)
    try {
      const updated = await api.save(job.id, item.id, { choice: decision, text, timings, revision: job.revision })
      if (updated.passage.timings?.length) priorTimings.current = updated.passage.timings
      setWording(updated.passage.selected_text)
      setChoice('suggestion')
      setTiming(null)
      onDirty(false)
      const template = updated.passage.timing_template
      if (repairAfterSave || (!timing && template?.length && wording !== item.selected_text)) {
        showTiming(updated.passage.selected_text, template?.length ? template : updated.passage.timings ?? [])
      }
      await onSave(updated, next)
    } catch (problem) {
      saveError(problem)
    } finally {
      onBusy(false)
    }
  }
  return (
    <article className="review-editor" onKeyDown={(event) => {
      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !event.repeat &&
          !event.nativeEvent.isComposing && event.nativeEvent.keyCode !== 229 &&
          (event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLInputElement)) {
        event.preventDefault()
        void save(true)
      }
    }}>
      <div className="passage-header">
        <h2 ref={heading} tabIndex={-1} className="mono">{clock(item.start_seconds)} – {clock(item.end_seconds)}</h2>
        {item.human_reviewed && !item.issues.length && <span className="status-chip">
          <Check size={14} aria-hidden="true" />{t('reviewedPassage')}</span>}
      </div>
      {item.issues.length > 0 && (
        <section className="review-note" aria-label={t('reason')}>
          <ul>
            {item.issues.map((issue) => {
              const range = clip ? issueRange(issue, clip) : undefined
              return <li key={issue.id}>
                <span className="issue-copy"><strong>{t(issue.category)}</strong> · {issueDescription(issue.reason, t) ?? issue.detail}</span>
                <div className="issue-actions">
                  {clip && <button type="button" className="button secondary issue-replay" disabled={!ready}
                    onClick={() => replayRange(range ?? clip)}>
                    <Play size={14} aria-hidden="true" />{t(range ? 'issueReplay' : 'passageReplay')}
                  </button>}
                  {(issue.category === 'coverage' || issue.category === 'seam') &&
                    <button type="button" className="button secondary issue-replay" disabled={!ready || busy || stale}
                      onClick={() => { void confirmIssue(issue) }}>{t('issueConfirm')}</button>}
                </div>
              </li>
            })}
          </ul>
        </section>
      )}
      <div className="review-audio">
        {clipError ? <div className="state-banner warning" role="alert">{clipError}
          <button type="button" className="text-button" onClick={() => setClipAttempt((value) => value + 1)}>{t('retry')}</button>
        </div> : clip && !playable ? <div className="audio-empty">{t('clipLoad')}</div>
          : <Player clip={playable} t={t} onPositionChange={reportPlayhead} replay={replay} speed={speed}
              onSpeedChange={onSpeedChange} onRetry={() => setClipAttempt((value) => value + 1)} />}
      </div>
      <div className="workspace-grid">
        <section className="editor-section">
          <h3>{t('accepted')}</h3>
          {timing ? (
            <div className="manual-timing">
              <p>{t('timingHint')}</p>
              {timing.map((cue, i) => (
                <fieldset className="timing-cue" key={cue.id}>
                  <legend>{t('cue')} {i + 1}{cue.needsConfirmation && ` · ${t('timingNeedsConfirmation')}`}</legend>
                  <label className="sr-only" htmlFor={`cue-${cue.id}`}>{t('accepted')}</label>
                  <textarea id={`cue-${cue.id}`} lang="zh-CN" value={cue.text} rows={2}
                    onChange={(e) => updateCue(cue.id, 'text', e.target.value)} disabled={busy} />
                  <div className="timing-fields">
                    {(['start', 'end'] as const).map((bound) => (
                      <div key={bound}>
                        <label htmlFor={`cue-${cue.id}-${bound}`}>{t(bound === 'start' ? 'startSeconds' : 'endSeconds')}</label>
                        <input id={`cue-${cue.id}-${bound}`} type="number" min={0} step={0.001}
                          value={cue[bound]} onChange={(e) => updateCue(cue.id, bound, e.target.value)} disabled={busy} />
                        <button type="button" className="text-button" disabled={!ready || busy}
                          aria-label={`${t('usePlayhead')} · ${t('cue')} ${i + 1} · ${t(bound === 'start' ? 'startSeconds' : 'endSeconds')}`}
                          onClick={() => usePlayhead(cue.id, bound)}>{t('usePlayhead')}</button>
                      </div>
                    ))}
                  </div>
                  {cue.start && cue.end && needsReadingTime(cue.text, Number(cue.end) - Number(cue.start)) &&
                    <p className="cue-readability">{t('timingReadability')}</p>}
                  <div className="passage-options">
                    <button type="button" className="text-button" disabled={!ready || busy}
                      onClick={() => {
                        const input = document.getElementById(`cue-${cue.id}`)
                        if (input instanceof HTMLTextAreaElement) splitCue(i, input.selectionStart)
                      }}>{t('timingSplit')}</button>
                    {i + 1 < timing.length && <button type="button" className="text-button" disabled={busy}
                      onClick={() => mergeCue(i)}>{t('timingMerge')}</button>}
                    {timing.length > 1 && <button type="button" className="text-button" disabled={busy}
                      onClick={() => setTiming(timing.filter((entry) => entry.id !== cue.id))}>{t('removeCue')}</button>}
                  </div>
                </fieldset>
              ))}
              <div className="passage-options">
                <button type="button" className="button secondary" disabled={busy} onClick={() => setTiming([
                  ...timing, { id: nextCue.current++, text: '', start: timing.at(-1)?.end ?? '', end: '' },
                ])}><Plus size={15} aria-hidden="true" />{t('addCue')}</button>
                <button type="button" className="text-button" disabled={busy} onClick={() => setTiming(null)}>{t('closeTiming')}</button>
              </div>
            </div>
          ) : (
            <>
              <div className="editor-card">
                <label className="sr-only" htmlFor="accepted-wording">{t('accepted')}</label>
                <textarea id="accepted-wording" lang="zh-CN" value={wording} spellCheck={false} disabled={busy}
                  onChange={(e) => { setWording(e.target.value); setChoice('edited') }} />
              </div>
              {item.timings?.some((cue) => needsReadingTime(cue.text, (cue.end_ms - cue.start_ms) / 1000)) &&
                <p className="cue-readability">{t('timingReadability')}</p>}
              <div className="passage-options">
                {wording.trim() && choice !== 'inaudible' && choice !== 'no_speech' &&
                  <button type="button" className="button secondary" onClick={openTiming} disabled={busy}>
                    <Clock3 size={15} aria-hidden="true" />{t('timingRepair')}
                  </button>}
                <details className="whole-passage-actions">
                  <summary>{t('wholePassageActions')}</summary>
                  <div className="passage-options">
                    <button type="button" className="button secondary" disabled={busy}
                      onClick={() => choose('suggestion', item.suggested_text)}>{t('suggestion')}</button>
                    <button type="button" className="button secondary" aria-pressed={choice === 'inaudible'} disabled={busy}
                      onClick={() => confirmWholePassage('inaudible', '［听不清］')}>{t('inaudible')}</button>
                    <button type="button" className="button secondary" aria-pressed={choice === 'no_speech'} disabled={busy}
                      onClick={() => confirmWholePassage('no_speech', '')}>{t('noSpeech')}</button>
                  </div>
                </details>
              </div>
            </>
          )}
          {error && <p className="notice error-text" role="alert">{error}</p>}
          {stale && <button type="button" className="button secondary" onClick={() => {
            if (!dirty || window.confirm(t('discardDraft'))) void onReload()
          }}>{t('reloadReview')}</button>}
          <div className="editor-actions">
            <button type="button" className="button secondary" disabled={saveDisabled}
              onClick={() => void save(false)}>{t(busy ? 'saving' : 'save')}</button>
            <button type="button" className="button primary" aria-keyshortcuts="Control+Enter Meta+Enter"
              disabled={saveDisabled}
              onClick={() => void save(true)}>{t('saveMore')}</button>
          </div>
        </section>
        <details className="compare-section" open={!compact}>
          <summary className="compare-toggle">{t('sources')}</summary>
          <div className="compare-content">
            {sources.map((source) => {
              const text = item.candidates?.[source]
              return (
                <div key={source} className={choice === source ? 'source-card source-chosen' : 'source-card'}>
                  <div className="source-card-head">{t(source)}</div>
                  <div className="source-text" lang={text ? 'zh-CN' : undefined}>
                    {text ? highlighted(text, item.candidate_spans?.[source] ?? [])
                      : source === 'openai' && item.unplaced_openai ? (
                        <><p className="small muted">{t('unplacedOpenai')}</p><span lang="zh-CN">{item.unplaced_openai}</span></>
                      ) : t(source === 'openai' && item.comparison_skipped ? 'comparisonSkipped' : 'noText')}
                  </div>
                  {text && <button type="button" className="source-use" aria-pressed={choice === source}
                    onClick={() => choose(source, text)} disabled={busy}>{t('useSource')} <ChevronRight size={14} aria-hidden="true" /></button>}
                </div>
              )
            })}
          </div>
        </details>
      </div>
    </article>
  )
}
