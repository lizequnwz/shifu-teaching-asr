import { useCallback, useEffect, useRef, useState } from 'react'
import { clock } from './Player'
import PassageEditor from './PassageEditor'
import { api, type Filter, type Job, type Passage, type ReviewPage, type SavedReview } from './api'
import type { Translate } from './i18n'

const filters: Filter[] = ['flagged', 'all', 'wording', 'coverage', 'timing', 'seam']
const compactReview = '(max-width: 760px)'

export default function ReviewPane({ job, t, onChange, onReload, dirty }: {
  job: Job
  t: Translate
  onChange: (job: Job) => void
  onReload: () => Promise<void>
  dirty: React.MutableRefObject<boolean>
}) {
  const [filter, setFilter] = useState<Filter>(job.status === 'needs_review' ? 'flagged' : 'all')
  const [page, setPage] = useState<ReviewPage | null>(null)
  const [selected, setSelected] = useState<Passage | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [notice, setNotice] = useState('')
  const [speed, setSpeed] = useState(1)
  const [compact, setCompact] = useState(() => window.matchMedia(compactReview).matches)
  const [queueOpen, setQueueOpen] = useState(false)
  const [sequence, setSequence] = useState(0)
  const heading = useRef<HTMLHeadingElement>(null)
  const requestedFocus = useRef(false)
  const onDirty = useCallback((value: boolean) => { dirty.current = value }, [dirty])
  const items = page?.passages ?? []
  const selection = page?.revision === job.revision ? selected ?? items[0] : undefined
  const cursor = page?.next_cursor

  useEffect(() => {
    const query = window.matchMedia(compactReview)
    const change = () => setCompact(query.matches)
    query.addEventListener('change', change)
    return () => query.removeEventListener('change', change)
  }, [])

  useEffect(() => {
    if (selection && requestedFocus.current) {
      requestedFocus.current = false
      heading.current?.focus()
    }
  }, [selection?.id])

  useEffect(() => {
    if (page?.revision === job.revision) return
    let active = true
    setLoading(true)
    setError('')
    void api.passages(job.id, filter).then((loaded) => {
      if (!active) return
      if (loaded.revision !== job.revision) {
        setError(t('stale'))
        return
      }
      setSelected(null)
      setPage(loaded)
    }, (problem) => {
      if (active) setError(problem instanceof Error ? problem.message : String(problem))
    }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [job.id, job.revision, filter, page?.revision, sequence, t])
  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => {
      if (dirty.current) event.preventDefault()
    }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [dirty])

  function switchTo(update: () => void) {
    if (saving || (dirty.current && !window.confirm(t('discardDraft')))) return
    dirty.current = false
    setNotice('')
    update()
  }

  function selectPassage(passage: Passage) {
    if (selection?.id === passage.id) {
      if (compact) {
        setQueueOpen(false)
        heading.current?.focus()
      }
      return
    }
    switchTo(() => {
      requestedFocus.current = true
      setSelected(passage)
      setQueueOpen(false)
    })
  }

  async function reload() {
    await onReload()
    setPage(null)
    setSelected(null)
    setSequence((value) => value + 1)
  }

  async function more() {
    if (!cursor || saving) return
    setLoading(true)
    setError('')
    try {
      const loaded = await api.passages(job.id, filter, cursor)
      if (loaded.revision !== job.revision) throw new Error(t('stale'))
      setPage((current) => current === page ? { ...loaded, passages: [...items, ...loaded.passages] } : current)
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : String(problem))
    } finally {
      setLoading(false)
    }
  }

  async function savePassage({ job: updated, passage }: SavedReview, next: boolean) {
    setError('')
    const unresolved = passage.issues.length > 0
    const inFilter = filter === 'all' || passage.issues.some((issue) => filter === 'flagged' || issue.category === filter)
    const rows = inFilter ? items.map((item) => item.id === passage.id ? passage : item) :
      items.filter((item) => item.id !== passage.id)
    const updatedPage = { passages: rows, revision: updated.revision ?? null, next_cursor: cursor ?? null }
    setPage(updatedPage)
    setSelected(passage)
    onChange(updated)

    if (unresolved || !next) {
      setNotice(t(unresolved ? 'reviewStillFlagged' : inFilter ? 'saved' : 'queueSavedOutsideFilter'))
      return
    }
    let following = rows.find((item) => item.start_seconds > passage.start_seconds)
    if (!following && cursor) {
      try {
        const loaded = await api.passages(job.id, filter, cursor)
        if (loaded.revision !== updated.revision) throw new Error(t('stale'))
        setPage({ ...loaded, passages: [...rows, ...loaded.passages] })
        following = loaded.passages[0]
      } catch (problem) {
        setError(problem instanceof Error ? problem.message : String(problem))
        setNotice(t('saved'))
        return
      }
    }
    if (following) {
      requestedFocus.current = true
      setSelected(following)
      setQueueOpen(false)
      setNotice(t('saved'))
    } else {
      setNotice(t('endOfQueue'))
    }
  }

  return (
    <div className="review-layout">
      <details className="review-queue" open={!compact || queueOpen}
        onToggle={(event) => { if (compact) setQueueOpen(event.currentTarget.open) }}>
        <summary className="review-queue-toggle">{t('queueToggle')}</summary>
        <aside className="passage-rail" aria-label={t('queueToggle')}>
          <div className="rail-heading"><h2>{t('passages')}</h2></div>
          <div className="rail-filter" role="group" aria-label={t('filter')}>
            {filters.map((key) => <button key={key} type="button" aria-pressed={filter === key}
              disabled={loading || saving} onClick={() => filter !== key && switchTo(() => {
                setFilter(key)
                setPage(null)
                setSelected(null)
              })}>{t(key)}</button>)}
          </div>
          <div className="passage-list" aria-busy={loading || saving}>
            {items.map((item) => <button key={item.id} type="button"
              className={selection?.id === item.id ? 'passage-card active' : 'passage-card'}
              aria-current={selection?.id === item.id ? 'true' : undefined} disabled={loading || saving}
              onClick={() => selectPassage(item)}>
              <span className="passage-time mono">{clock(item.start_seconds)}</span>
              <span className="passage-excerpt" lang={item.selected_text ? 'zh-CN' : undefined}>{item.selected_text || t('noText')}</span>
            </button>)}
            {!items.length && !loading && !error && <p className="small muted">{t('noFlags')}</p>}
          </div>
          {cursor && <button type="button" className="button secondary" disabled={loading || saving}
            onClick={() => void more()}>{t('more')}</button>}
        </aside>
      </details>
      <div>
        <p className="save-announcement" role="status">{notice}</p>
        {error && <div className="state-banner warning" role="alert">{error}
          <button type="button" className="text-button" onClick={() => {
            if (!dirty.current || window.confirm(t('discardDraft'))) void reload()
          }}>{t('reloadReview')}</button>
        </div>}
        {selection ? (
          <PassageEditor key={`${selection.id}:${sequence}`} job={job} item={selection} t={t} onDirty={onDirty}
            compact={compact} speed={speed} onSpeedChange={setSpeed} heading={heading}
            busy={saving || loading} onBusy={setSaving} onReload={reload} onSave={savePassage} />
        ) : (
          <div className="empty-card">{t(loading ? 'loading' : 'noFlags')}</div>
        )}
      </div>
    </div>
  )
}
