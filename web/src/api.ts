export type Clip = { id: string; start_seconds: number; end_seconds: number; url?: string }
export type PlayableClip = Clip & { url: string }
export const sources = ['gemini', 'openai'] as const
export type Source = (typeof sources)[number]
export type Choice = Source | 'suggestion' | 'edited' | 'inaudible' | 'no_speech'
export type Timing = { text: string; start_ms: number; end_ms: number }
export type Category = 'wording' | 'coverage' | 'timing' | 'seam'
export type Filter = 'all' | 'flagged' | Category
export type Issue = { category: Category; reason: string; detail: string; start_sample?: number; end_sample?: number }
export type Passage = {
  id: string
  start_seconds: number
  end_seconds: number
  clips: Clip[]
  issues: Issue[]
  candidates?: Partial<Record<Source, string>>
  candidate_spans?: Partial<Record<Source, { start: number; end: number }[]>>
  selected_text: string
  suggested_text: string
  unplaced_openai?: string | null
  timings?: Timing[]
  human_reviewed?: boolean
}
export type Status = 'uploading' | 'queued' | 'processing' | 'needs_review' | 'ready_to_publish' |
  'published' | 'paused' | 'failed' | 'no_speech'
export type Job = {
  id: string
  name: string
  status: Status
  revision?: string
  published_revision?: string
  error?: string
  retry_request?: string
  upload_bytes?: number
  summary?: {
    duration_seconds?: number
    audio_blocks?: number
    needs_listening?: number
    listening_seconds?: number
  }
  operation?: { stage: string; completed?: number; total?: number }
}
export type Config =
  | { mode: 'local'; token: string }
  | { mode: 'online'; supabase_url: string; supabase_publishable_key: string; allowed_regions?: string[]; max_upload_bytes: number }
type Page<T, K extends string> = Record<K, T[]> & { next_cursor: string | null }
export type ReviewPage = Page<Passage, 'passages'> & { revision: string | null }
export type SavedReview = { job: Job; passage: Passage }
type UploadCredentials = { endpoint: string; bucket: string; object_key: string; token: string }
type Decision = { choice: Choice; text?: string; timings?: Timing[]; revision: string }
export type Artifact = 'txt' | 'srt'

export class ApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message)
  }
}
let localToken: string | undefined
let accessToken: string | undefined

export function setAccessToken(token?: string) {
  accessToken = token
}

async function request<T>(url: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (accessToken) headers.set('Authorization', `Bearer ${accessToken}`)
  if (localToken && init.method && init.method !== 'GET') headers.set('X-Subtitle-Token', localToken)
  const res = await fetch(url, { ...init, headers })
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    const message = typeof body?.detail === 'string' ? body.detail : res.statusText
    throw new ApiError(res.status, message)
  }
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

function json(method: 'POST' | 'PUT', body: object): RequestInit {
  return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
}

const jobPath = (id: string) => `/api/jobs/${encodeURIComponent(id)}`

export const api = {
  config: async () => {
    const config = await request<Config>('/api/config')
    localToken = config.mode === 'local' ? config.token : undefined
    return config
  },
  jobs: (cursor?: string) => request<Page<Job, 'jobs'>>(
    `/api/jobs${cursor ? `?cursor=${encodeURIComponent(cursor)}` : ''}`,
  ),
  job: (id: string) => request<Job>(jobPath(id)),
  createLocal: (file: File, terms: string[]) => {
    const data = new FormData()
    data.set('file', file)
    data.set('terms', terms.join('\n'))
    return request<Job>('/api/jobs', { method: 'POST', body: data })
  },
  createOnline: (file: File, terms: string[]) => request<{ job: Job; upload: UploadCredentials }>(
    '/api/jobs', json('POST', { name: file.name, bytes: file.size, mime_type: file.type || 'application/octet-stream', terms }),
  ),
  uploadToken: (id: string) => request<UploadCredentials>(`${jobPath(id)}/upload-token`, json('POST', {})),
  completeUpload: (id: string) => request<Job>(`${jobPath(id)}/upload-complete`, json('POST', {})),
  passages: (id: string, filter: Filter, cursor?: string) => {
    const query = new URLSearchParams({ filter })
    if (cursor) query.set('cursor', cursor)
    return request<ReviewPage>(`${jobPath(id)}/passages?${query}`)
  },
  clip: async (id: string, clip: Clip): Promise<PlayableClip> =>
    clip.url ? { ...clip, url: clip.url } : {
      ...clip,
      ...await request<{ url: string }>(`${jobPath(id)}/clips/${encodeURIComponent(clip.id)}`),
    },
  save: (id: string, passage: string, body: Decision) => request<SavedReview>(
    `${jobPath(id)}/decisions/${encodeURIComponent(passage)}`, json('PUT', body),
  ),
  publish: (id: string, revision: string) => request<Job>(`${jobPath(id)}/publish`, json('POST', { revision })),
  resume: (id: string, retry_request?: string) => request<Job>(
    `${jobPath(id)}/resume`, json('POST', retry_request ? { retry_request } : {}),
  ),
  export: (id: string, revision: string, format: Artifact) => request<{ url: string }>(
    `${jobPath(id)}/exports/${encodeURIComponent(revision)}/${format}`,
  ),
  remove: (id: string) => request<void>(jobPath(id), { method: 'DELETE' }),
}

export async function uploadFile(
  id: string, file: File, credentials: UploadCredentials, publishableKey: string, onProgress: (value: number) => void,
) {
  const { Upload } = await import('tus-js-client')
  return new Promise<void>((resolve, reject) => {
    const upload = new Upload(file, {
      endpoint: credentials.endpoint,
      headers: { apikey: publishableKey, 'x-signature': credentials.token },
      uploadDataDuringCreation: true,
      removeFingerprintOnSuccess: true,
      chunkSize: 6 * 1024 * 1024,
      retryDelays: [0, 1000, 3000, 5000, 10000],
      metadata: {
        bucketName: credentials.bucket,
        objectName: credentials.object_key,
        contentType: file.type || 'application/octet-stream',
        cacheControl: '0',
      },
      fingerprint: async () => JSON.stringify(['subtitle-studio', id, file.name, file.size, file.lastModified]),
      onProgress: (sent, total) => onProgress(total ? sent / total : 0),
      onSuccess: () => resolve(),
      onError: reject,
    })
    upload.findPreviousUploads().then((previous) => {
      if (previous.length) upload.resumeFromPreviousUpload(previous[0])
      upload.start()
    }, reject)
  })
}
