import { lazy, Suspense, useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import * as Dialog from '@radix-ui/react-dialog'
import { ArrowLeft, ArrowRight, AudioLines, Download, FolderOpen, Plus, Trash2, X } from 'lucide-react'
import ReviewPane from './ReviewPane'
import { api, uploadFile, type Artifact, type Config, type Job } from './api'
import { en, zh, type Key, type Lang, type Translate } from './i18n'
import { clock } from './Player'

const Account = lazy(() => import('./Account'))

const errorMessage = (problem: unknown) => problem instanceof Error ? problem.message : String(problem)
const active = (job: Job) => job.status === 'queued' || job.status === 'processing'
const stageKeys: Record<string, Key> = {
  preflight: 'probe', probe: 'probe', audio_preflight: 'probe', prepare: 'prepare', decode: 'decode',
  transcribe: 'transcribe', cloud_draft: 'transcribe', reconcile: 'reconcile', review: 'review_stage',
  preparing_review: 'review_stage',
}
const shortClock = (seconds: number) => clock(seconds).split('.')[0]
const fileTypes = '.m4a,.mp3,.wav,.flac,.aac,.mp4,.mov,.webm,.ogg,.aiff,.aif,.opus'
const localUploadBytes = 512 * 1024 * 1024

function UploadDialog({ config, pending, t, close, added, complete }: {
  config: Config
  pending?: Job
  t: Translate
  close: () => void
  added: (job: Job) => void
  complete: (job: Job) => void
}) {
  const [target, setTarget] = useState(pending)
  const [file, setFile] = useState<File | null>(null)
  const [terms, setTerms] = useState('')
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState<number | undefined>()
  const [error, setError] = useState('')
  const opener = useRef(document.activeElement)
  const maxUploadBytes = config.mode === 'online' ? config.max_upload_bytes : localUploadBytes
  const uploadLimit = config.mode === 'online'
    ? t('onlineUploadLimit').replace('{size}', `${maxUploadBytes / 1_000_000} MB`)
    : t('uploadLimit')
  const isTooLarge = file !== null && file.size > maxUploadBytes

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!file) {
      setError(t('selectAudio'))
      return
    }
    if (isTooLarge) return
    if (target && (file.name.trim() !== target.name || (target.upload_bytes != null && file.size !== target.upload_bytes))) {
      setError(t('uploadMismatch'))
      return
    }
    setError('')
    setProgress(undefined)
    setBusy(true)
    try {
      const vocabulary = terms.split(/\r?\n/).map((term) => term.trim()).filter(Boolean)
      if (config.mode === 'local') {
        complete(await api.createLocal(file, vocabulary))
      } else {
        const entry = target ? { job: target, upload: await api.uploadToken(target.id) }
          : await api.createOnline(file, vocabulary)
        setTarget(entry.job)
        added(entry.job)
        await uploadFile(entry.job.id, file, entry.upload, config.supabase_publishable_key, setProgress)
        complete(await api.completeUpload(entry.job.id))
      }
    } catch (problem) {
      setError(errorMessage(problem))
    } finally {
      setBusy(false)
    }
  }

  async function finishPreviousUpload() {
    if (!target) return
    setError('')
    setProgress(undefined)
    setBusy(true)
    try {
      complete(await api.completeUpload(target.id))
    } catch (problem) {
      setError(errorMessage(problem))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog.Root open onOpenChange={(open) => { if (!open && !busy) close() }}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="dialog-content" onCloseAutoFocus={(event) => {
          if (opener.current instanceof HTMLElement && opener.current.isConnected) {
            event.preventDefault()
            opener.current.focus()
          }
        }}>
          <div className="dialog-top">
            <Dialog.Title className="dialog-title">{t(target ? 'uploadResume' : 'newJob')}</Dialog.Title>
            <Dialog.Close className="icon-btn" aria-label={t('close')} disabled={busy}><X size={20} aria-hidden="true" /></Dialog.Close>
          </div>
          <Dialog.Description className="dialog-description">
            {t(target ? 'uploadChooseSame' : config.mode === 'local' ? 'localPrivacy' : 'onlinePrivacy')}
          </Dialog.Description>
          <form className="upload-form" onSubmit={submit}>
            <label className="field">
              <span>{t('audioFile')}</span>
              <input className="file-field" type="file" accept={fileTypes} onChange={(event) => {
                setFile(event.target.files?.[0] ?? null)
                setError('')
              }} aria-describedby="upload-limit" required disabled={busy} />
            </label>
            <p className="small muted" id="upload-limit">{uploadLimit}</p>
            {!target && (
              <label className="field">
                <span>{t('confirmedTerms')}</span>
                <textarea rows={4} value={terms} onChange={(event) => setTerms(event.target.value)}
                  placeholder={t('termsPlaceholder')} aria-describedby="terms-help" disabled={busy} />
                <small id="terms-help">{t('termsHint')}</small>
              </label>
            )}
            {busy && <progress className="progress-track" aria-label={t('uploadProgress')} max={1} value={progress} />}
            {(isTooLarge || error) && <p className="form-error" role="alert">
              {isTooLarge ? t('uploadTooLarge') : error}
            </p>}
            <div className="dialog-actions">
              {!busy && target && <button className="button secondary" type="button"
                onClick={() => void finishPreviousUpload()}>{t('uploadRetry')}</button>}
              <button type="submit" className="button primary" disabled={busy || isTooLarge}>
                {t(busy ? 'uploading' : target ? 'uploadResume' : 'uploadRecording')}
              </button>
            </div>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}

function JobPage({ initial, id, t, onBack, onResumeUpload, onChange, onRemove }: {
  initial?: Job
  id: string
  t: Translate
  onBack: () => void
  onResumeUpload: (job: Job) => void
  onChange: (job: Job) => void
  onRemove: (id: string) => void
}) {
  const [job, setJob] = useState(initial)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const dirty = useRef(false)
  const update = useCallback((entry: Job) => {
    setJob(entry)
    onChange(entry)
  }, [onChange])
  const refresh = useCallback(async () => {
    try {
      update(await api.job(id))
      setError('')
    } catch (problem) {
      setError(errorMessage(problem))
    }
  }, [id, update])
  useEffect(() => { void refresh() }, [refresh])
  const processing = job && active(job)
  useEffect(() => {
    if (!processing) return
    const timer = window.setInterval(() => { void refresh() }, 3500)
    return () => window.clearInterval(timer)
  }, [processing, refresh])

  async function run(action: () => Promise<void>) {
    setBusy(true)
    setError('')
    try {
      await action()
    } catch (problem) {
      setError(errorMessage(problem))
    } finally {
      setBusy(false)
    }
  }
  function back() {
    if (!dirty.current || window.confirm(t('discardDraft'))) onBack()
  }
  async function download(format: Artifact) {
    if (!job?.published_revision) return
    await run(async () => {
      const { url } = await api.export(id, job.published_revision!, format)
      const link = document.createElement('a')
      link.href = url
      link.download = ''
      link.click()
    })
  }
  if (!job) return (
    <div className="page">
      <button type="button" className="back-link" onClick={back}><ArrowLeft size={15} aria-hidden="true" />{t('back')}</button>
      {error ? <div className="state-banner warning" role="alert">
        {error} <button className="button secondary" type="button" onClick={() => void refresh()}>{t('retry')}</button>
      </div> : <p>{t('loading')}</p>}
    </div>
  )
  const operation = job.operation
  const progressLabel = t(stageKeys[operation?.stage ?? ''] ?? 'working')
  return (
    <div className="page review-page">
      <header className="review-heading">
        <div>
          <button type="button" className="back-link" onClick={back}><ArrowLeft size={15} aria-hidden="true" />{t('back')}</button>
          <h1>{job.name}</h1>
          {job.summary?.duration_seconds != null && <p className="mono muted">{shortClock(job.summary.duration_seconds)}</p>}
        </div>
        <div className="job-actions">
          <span className={`status-chip status-${job.status}`} role="status">{t(job.status)}</span>
          {!processing &&
            <button type="button" className="button secondary" disabled={busy} onClick={() => {
              if (window.confirm(t('deleteConfirmation'))) void run(async () => {
                await api.remove(id)
                onRemove(id)
              })
            }}><Trash2 size={15} aria-hidden="true" />{t('deleteJob')}</button>}
        </div>
      </header>
      {error && <div className="state-banner warning" role="alert">{error}</div>}
      {processing && <section className="operation-card" aria-label={progressLabel}>
        <div className="operation-main">
          <strong>{job.status === 'queued' ? t('queued') : progressLabel}</strong>
          <progress className="progress-track" aria-label={progressLabel}
            max={operation?.total || undefined} value={operation?.total ? operation.completed : undefined} />
        </div>
        {operation?.total && operation.completed != null
          ? <span className="mono">{operation.completed} / {operation.total}</span> : null}
      </section>}
      {job.status === 'uploading' && <div className="state-banner">
        <p>{t('uploadChooseSame')}</p>
        <button type="button" className="button primary" onClick={() => onResumeUpload(job)}>{t('uploadResume')}</button>
      </div>}
      {(job.status === 'paused' || job.status === 'failed') && (
        <div className="state-banner warning">
          <p>{job.error || t(job.status)}</p>
          {job.status === 'paused' && <button type="button" className="button secondary" disabled={busy} onClick={() => {
            if (!job.retry_request || window.confirm(t('retryConfirmation')))
              void run(async () => update(await api.resume(id, job.retry_request)))
          }}>{t(job.retry_request ? 'retryRequest' : 'resume')}</button>}
        </div>
      )}
      {job.status === 'no_speech' && <div className="empty-card">{t('no_speech')}</div>}
      {job.status === 'ready_to_publish' && job.revision && (
        <div className="state-banner">
          <strong>{t('ready_to_publish')}</strong>
          <button type="button" className="button primary" disabled={busy} onClick={() => {
            if (dirty.current) {
              setError(t('discardDraft'))
              return
            }
            void run(async () => update(await api.publish(id, job.revision!)))
          }}>{t(busy ? 'publishing' : 'publish')}</button>
        </div>
      )}
      {job.published_revision && (
        <div className="state-banner">
          <strong>{t('publishedResult')}</strong>
          <div className="job-actions">
            <button type="button" className="button secondary" disabled={busy} onClick={() => void download('txt')}>
              <Download size={15} aria-hidden="true" />{t('downloadTxt')}
            </button>
            <button type="button" className="button secondary" disabled={busy} onClick={() => void download('srt')}>
              <Download size={15} aria-hidden="true" />{t('downloadSrt')}
            </button>
          </div>
        </div>
      )}
      {job.revision && job.summary?.audio_blocks != null && (
        <>
          <p className="review-totals">
            {job.summary?.needs_listening ?? 0} {t('issueCount')}
            {job.summary?.listening_seconds != null && <> · {shortClock(job.summary.listening_seconds)} {t('listeningTime')}</>}
          </p>
          <ReviewPane job={job} t={t} onChange={update} onReload={refresh} dirty={dirty} />
        </>
      )}
    </div>
  )
}

function Workspace({ config, t, email, signOut }: {
  config: Config
  t: Translate
  email?: string
  signOut?: () => Promise<void>
}) {
  const [id, setId] = useState(() => new URLSearchParams(location.search).get('job'))
  const [jobs, setJobs] = useState<Job[]>([])
  const [cursor, setCursor] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState('')
  const [upload, setUpload] = useState<{ pending?: Job } | null>(null)
  const [jobGeneration, setJobGeneration] = useState(0)
  const remember = useCallback((job: Job) => setJobs((current) =>
    current.some((entry) => entry.id === job.id)
      ? current.map((entry) => entry.id === job.id ? job : entry)
      : [job, ...current],
  ), [])

  const listJobs = useCallback(async (after?: string) => {
    setLoadError('')
    setLoading(true)
    try {
      const page = await api.jobs(after)
      setJobs((current) => after ? [...current, ...page.jobs] : page.jobs)
      setCursor(page.next_cursor)
    } catch (problem) {
      setLoadError(errorMessage(problem))
    } finally {
      setLoading(false)
    }
  }, [])
  useEffect(() => { if (!id) void listJobs() }, [id, listJobs])

  function navigate(job: string | null) {
    const url = new URL(location.href)
    if (job) url.searchParams.set('job', job)
    else url.searchParams.delete('job')
    history.replaceState(null, '', url)
    setId(job)
  }

  return (
    <>
      <header className="topbar studio-header">
        <div className="brand">
          <span className="brand-mark"><AudioLines size={21} aria-hidden="true" /></span>
          <div><div className="brand-title">{t('app')}</div><div className="mode-name">{t(config.mode)}</div></div>
        </div>
        <div className="top-actions">
          {email && <span className="studio-account-name small muted">{email}</span>}
          {signOut && <button className="button secondary" type="button" onClick={() => void signOut().catch(
            (problem) => setLoadError(errorMessage(problem)),
          )}>{t('signOut')}</button>}
        </div>
      </header>
      <main className="main">
        {loadError && <div className="global-error" role="alert">
          {loadError}<button type="button" className="text-button" onClick={() => void listJobs()}>{t('retry')}</button>
        </div>}
        {id ? (
          <JobPage key={`${id}:${jobGeneration}`} id={id} initial={jobs.find((job) => job.id === id)} t={t} onBack={() => navigate(null)}
            onChange={remember} onResumeUpload={(pending) => setUpload({ pending })}
            onRemove={(removed) => { setJobs((current) => current.filter((job) => job.id !== removed)); navigate(null) }} />
        ) : (
          <div className="page library-page">
            <section className="library-hero">
              <div className="hero-copy">
                <h1>{t('recordingsLead')}</h1>
                <p>{t('recordingsBody')}</p>
                <button type="button" className="button primary hero-button" onClick={() => setUpload({})}>
                  <Plus size={18} aria-hidden="true" />{t('newJob')}
                </button>
              </div>
              <div className="hero-art" aria-hidden="true">
                <div className="hero-art-grid" />
                <div className="art-disc">
                  <div className="disc-ring ring-one" /><div className="disc-ring ring-two" />
                  <div className="disc-center"><AudioLines size={30} /></div>
                </div>
              </div>
            </section>
            <section className="projects-section" aria-busy={loading}>
              <div className="section-heading"><h2>{t('library')}</h2></div>
              {jobs.length ? (
                <div className="job-list">
                  {jobs.map((job) => (
                    <button className="job-row" key={job.id} type="button" onClick={() => navigate(job.id)}>
                      <span className="job-art"><AudioLines size={20} aria-hidden="true" /></span>
                      <span className="job-main">
                        <strong>{job.name}</strong>
                        {job.summary?.duration_seconds != null && <span>{shortClock(job.summary.duration_seconds)}</span>}
                      </span>
                      <span className={`status-chip status-${job.status}`}>{t(job.status)}</span>
                      <ArrowRight size={17} aria-hidden="true" />
                    </button>
                  ))}
                </div>
              ) : loading ? <div className="empty-card">{t('loading')}</div> : (
                <div className="empty-card">
                  <FolderOpen size={24} aria-hidden="true" /><h3>{t('noJobs')}</h3><p>{t('noJobsBody')}</p>
                </div>
              )}
              {cursor && <button type="button" className="button secondary" disabled={loading}
                onClick={() => void listJobs(cursor)}>{t('more')}</button>}
            </section>
            <footer className="studio-footer">
              <p>{t(config.mode === 'local' ? 'localPrivacy' : 'onlinePrivacy')}</p>
              {config.mode === 'online' && <details><summary>{t('localSetup')}</summary><p>{t('localInstructions')}</p></details>}
            </footer>
          </div>
        )}
      </main>
      {upload && <UploadDialog config={config} t={t} pending={upload.pending} close={() => {
        setUpload(null)
        if (!id) void listJobs()
      }} added={remember}
        complete={(job) => {
          remember(job)
          setUpload(null)
          if (job.id === id) setJobGeneration((value) => value + 1)
          navigate(job.id)
        }} />}
    </>
  )
}

export default function App() {
  const [lang, setLang] = useState<Lang>(() => localStorage.getItem('subtitle-studio-language') === 'en' ? 'en' : 'zh')
  const [config, setConfig] = useState<Config | null>(null)
  const [error, setError] = useState('')
  const t = useCallback((key: Key) => (lang === 'zh' ? zh : en)[key], [lang])
  const load = useCallback(async () => {
    setError('')
    try {
      setConfig(await api.config())
    } catch (problem) {
      setError(errorMessage(problem))
    }
  }, [])
  useEffect(() => { void load() }, [load])
  useEffect(() => {
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en'
    document.title = t('app')
    localStorage.setItem('subtitle-studio-language', lang)
  }, [lang, t])
  return (
    <>
      <label className="site-language">
        <span className="sr-only">{t('language')}</span>
        <select value={lang} onChange={(event) => setLang(event.target.value as Lang)}>
          <option value="zh">简体中文</option>
          <option value="en">English</option>
        </select>
      </label>
      {!config ? <main className="center-state">
        {error ? <><p role="alert">{error}</p><button type="button" className="button secondary" onClick={() => void load()}>{t('retry')}</button></>
          : <p>{t('loading')}</p>}
      </main> : config.mode === 'local' ? <Workspace config={config} t={t} /> : (
        <Suspense fallback={<main className="center-state"><p>{t('loading')}</p></main>}>
          <Account config={config} t={t} lang={lang}>
            {(email, signOut) => <Workspace key={email} config={config} t={t} email={email} signOut={signOut} />}
          </Account>
        </Suspense>
      )}
    </>
  )
}
