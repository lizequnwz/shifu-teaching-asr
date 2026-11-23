import { useCallback, useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import {
  Pause,
  Play,
  RotateCcw,
  RotateCw,
  Repeat2,
  Volume2,
} from 'lucide-react'
import type { PlayableClip } from './api'
import type { Key } from './i18n'

export type PlaybackRange = { start_seconds: number; end_seconds: number }
export type ReplayRequest = PlaybackRange & { id: number }

type ClipRange = { start: number; end: number }
type Props = {
  clip: PlayableClip | null
  t: (key: Key) => string
  onPositionChange: (url: string, seconds: number | null) => void
  replay?: ReplayRequest
  speed: number
  onSpeedChange: (speed: number) => void
  onRetry: () => void
}
export const clock = (s: number) => {
  if (!Number.isFinite(s)) return '00:00.000'
  const ms = Math.max(0, Math.round(s * 1000))
  const seconds = Math.floor(ms / 1000)
  const hours = Math.floor(seconds / 3600)
  const minutes = String(Math.floor(seconds / 60) % 60).padStart(2, '0')
  return `${hours ? `${hours}:` : ''}${minutes}:${String(seconds % 60).padStart(2, '0')}.${String(ms % 1000).padStart(3, '0')}`
}
export default function Player({ clip, t, onPositionChange, replay, speed, onSpeedChange, onRetry }: Props) {
  const container = useRef<HTMLDivElement>(null)
  const wave = useRef<WaveSurfer | null>(null)
  const looping = useRef(false)
  const playbackRange = useRef<ClipRange | null>(null)
  const playedRequest = useRef<number | null>(null)
  const [readyUrl, setReadyUrl] = useState<string | null>(null)
  const [playing, setPlaying] = useState(false)
  const [pos, setPos] = useState(0)
  const [duration, setDuration] = useState(0)
  const [loop, setLoop] = useState(false)
  const [error, setError] = useState(false)
  const [startRequired, setStartRequired] = useState(false)
  const [range, setRange] = useState<ClipRange | null>(null)
  const ready = Boolean(clip && readyUrl === clip.url)
  const clipUrl = clip?.url
  const clipStart = clip?.start_seconds

  const selectRange = useCallback((selected: ClipRange | null) => {
    playbackRange.current = selected
    setRange(selected)
  }, [])
  const play = useCallback((player: WaveSurfer, start?: number) => {
    setStartRequired(false)
    void player.play(start).catch((problem: unknown) => {
      if (wave.current !== player) return
      if (problem instanceof DOMException && problem.name === 'NotAllowedError') {
        setStartRequired(true)
      } else if (!(problem instanceof DOMException && problem.name === 'AbortError')) {
        setReadyUrl(null)
        setError(true)
      }
    })
  }, [])

  useEffect(() => {
    if (!clipUrl || clipStart === undefined || !container.current) return
    setReadyUrl(null)
    setPlaying(false)
    setError(false)
    setStartRequired(false)
    setPos(0)
    setDuration(0)
    selectRange(null)
    const w = WaveSurfer.create({
      container: container.current,
      url: clipUrl,
      waveColor: '#b7c0d0',
      progressColor: '#3156b6',
      cursorColor: '#244ca8',
      barWidth: 2,
      barGap: 3,
      barRadius: 2,
      height: 62,
      normalize: true,
      dragToSeek: true,
    })
    wave.current = w
    w.on('ready', (d) => {
      setDuration(d)
      setReadyUrl(clipUrl)
    })
    w.on('timeupdate', (s) => setPos(s))
    w.on('play', () => {
      setPlaying(true)
      setStartRequired(false)
    })
    w.on('pause', () => setPlaying(false))
    w.on('interaction', () => selectRange(null))
    w.on('audioprocess', (time) => {
      const selected = playbackRange.current
      if (!selected || time < selected.end) return
      if (looping.current) {
        w.setTime(selected.start)
      } else {
        w.pause()
        w.setTime(selected.end)
      }
    })
    w.on('finish', () => {
      setPlaying(false)
      if (looping.current) play(w, playbackRange.current?.start ?? 0)
    })
    w.on('error', () => {
      setReadyUrl(null)
      setStartRequired(false)
      setError(true)
    })
    return () => {
      wave.current = null
      w.destroy()
    }
  }, [clipUrl, clipStart, play, selectRange])
  useEffect(() => {
    if (ready) wave.current?.setPlaybackRate(speed, true)
  }, [ready, speed])
  useEffect(() => {
    if (!replay) {
      selectRange(null)
      return
    }
    const player = wave.current
    if (!ready || !player || clipStart === undefined || playedRequest.current === replay.id) return
    const start = Math.max(0, replay.start_seconds - clipStart)
    const end = Math.min(duration, replay.end_seconds - clipStart)
    if (!(start < end)) return
    playedRequest.current = replay.id
    selectRange({ start, end })
    play(player, start)
  }, [replay, ready, duration, clipStart, play, selectRange])
  useEffect(() => {
    if (clipUrl && clipStart !== undefined)
      onPositionChange(clipUrl, ready ? clipStart + pos : null)
  }, [clipUrl, clipStart, onPositionChange, pos, ready])
  useEffect(() => {
    if (clipUrl) return () => onPositionChange(clipUrl, null)
  }, [clipUrl, onPositionChange])
  if (!clip)
    return (
      <div className="audio-empty">
        <Volume2 size={18} aria-hidden="true" />
        {t('noAudio')}
      </div>
    )
  const seek = (time: number) => {
    if (!ready || !wave.current) return
    selectRange(null)
    wave.current.setTime(Math.max(0, Math.min(duration, time)))
  }
  const move = (delta: number) => {
    if (wave.current) seek(wave.current.getCurrentTime() + delta)
  }
  const togglePlayback = () => {
    const player = wave.current
    if (!ready || !player) return
    if (player.isPlaying()) {
      player.pause()
      return
    }
    const selected = playbackRange.current
    const time = player.getCurrentTime()
    const restartRange = selected && (time < selected.start || time >= selected.end - 0.001)
    play(player, restartRange ? selected.start : undefined)
  }
  const position = clip.start_seconds + (ready ? pos : 0)
  return (
    <div className="player">
      <div className="wave-head">
        <span className="eyebrow">{t('interval')}</span>
        <span className="mono small muted">
          {clock(clip.start_seconds)} — {clock(clip.end_seconds)}
        </span>
      </div>
      <div className="wave-wrap">
        <div ref={container} aria-hidden="true" />
        {ready && range && (
          <div className="wave-range" aria-hidden="true" style={{
            left: `${100 * range.start / duration}%`,
            width: `${100 * (range.end - range.start) / duration}%`,
          }} />
        )}
        {!ready && !error && (
          <div className="wave-loading">{t('loadingAudio')}</div>
        )}
        {error && (
          <div className="wave-loading wave-error error-text">
            <span role="alert">{t('audioError')}</span>
            <button type="button" className="text-button" onClick={onRetry}>{t('retry')}</button>
          </div>
        )}
      </div>
      {ready && range && (
        <p className="wave-range-summary">
          {t('activePlaybackRange')} <span className="mono">
            {clock(clip.start_seconds + range.start)} — {clock(clip.start_seconds + range.end)}
          </span>
        </p>
      )}
      {startRequired && <p className="wave-range-summary" role="status">{t('startPlayback')}</p>}
      <input
        className="seek"
        type="range"
        aria-label={t('seek')}
        aria-valuetext={ready ? `${position.toFixed(3)} ${t('recordingSeconds')}` : undefined}
        min="0"
        max={duration || 1}
        step="0.01"
        value={ready ? Math.min(pos, duration) : 0}
        disabled={!ready}
        onChange={(e) => seek(Number(e.target.value))}
      />
      <div className="player-controls">
        <div className="transport">
          <button
            type="button"
            className="icon-btn"
            aria-label={t('back5')}
            title={t('back5')}
            disabled={!ready}
            onClick={() => move(-5)}
          >
            <RotateCcw size={17} aria-hidden="true" />
            <span className="skip-num" aria-hidden="true">5</span>
          </button>
          <button
            type="button"
            className="play-btn"
            aria-label={playing ? t('pause') : t('play')}
            disabled={!ready}
            onClick={togglePlayback}
          >
            {playing ? (
              <Pause size={18} fill="currentColor" aria-hidden="true" />
            ) : (
              <Play size={18} fill="currentColor" aria-hidden="true" />
            )}
          </button>
          <button
            type="button"
            className="icon-btn"
            aria-label={t('forward5')}
            title={t('forward5')}
            disabled={!ready}
            onClick={() => move(5)}
          >
            <RotateCw size={17} aria-hidden="true" />
            <span className="skip-num" aria-hidden="true">5</span>
          </button>
          <span className="time-display mono">
            {clock(position)}{' '}
            <span className="time-divider">/</span> {clock(clip.end_seconds)}
          </span>
        </div>
        <div className="player-settings">
          <button
            type="button"
            className={`icon-btn loop-btn ${loop ? 'on' : ''}`}
            aria-label={t('loop')}
            aria-pressed={loop}
            title={t('loop')}
            disabled={!ready}
            onClick={() => {
              looping.current = !loop
              setLoop(!loop)
            }}
          >
            <Repeat2 size={18} aria-hidden="true" />
          </button>
          <label className="speed-label">
            <span className="sr-only">{t('playbackSpeed')}</span>
            <select
              value={speed}
              disabled={!ready}
              onChange={(e) => onSpeedChange(Number(e.target.value))}
            >
              <option value="0.75">0.75×</option>
              <option value="1">1×</option>
              <option value="1.25">1.25×</option>
              <option value="1.5">1.5×</option>
            </select>
          </label>
        </div>
      </div>
    </div>
  )
}
