import { useEffect, useState, type FormEvent, type ReactNode } from 'react'
import { createClient, type Session } from '@supabase/supabase-js'
import { setAccessToken, type Config } from './api'
import type { Lang, Translate } from './i18n'

type Props = {
  config: Extract<Config, { mode: 'online' }>
  t: Translate
  lang: Lang
  children: (email: string, signOut: () => Promise<void>) => ReactNode
}

export default function Account({ config, t, lang, children }: Props) {
  const [client] = useState(() => createClient(config.supabase_url, config.supabase_publishable_key))
  const [session, setSession] = useState<Session | null | undefined>(undefined)
  const [email, setEmail] = useState('')
  const [country, setCountry] = useState('')
  const [adult, setAdult] = useState(false)
  const [code, setCode] = useState('')
  const [codeSent, setCodeSent] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const regions = config.allowed_regions ?? []
  const countries = new Intl.DisplayNames([lang === 'zh' ? 'zh-CN' : 'en'], { type: 'region' })
  const profile = session?.user.user_metadata
  const needsProfile = !!session && !(profile?.adult === true && regions.includes(profile.country))
  useEffect(() => {
    const { data: { subscription } } = client.auth.onAuthStateChange((_event, value) => {
      setAccessToken(value?.access_token)
      setSession(value)
    })
    return () => {
      subscription.unsubscribe()
      setAccessToken(undefined)
    }
  }, [client])

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')
    setBusy(true)
    try {
      const result = needsProfile
        ? await client.auth.updateUser({ data: { country, adult } })
        : codeSent
          ? await client.auth.verifyOtp({ email, token: code.trim(), type: 'email' })
          : await client.auth.signInWithOtp({
            email: email.trim(),
            options: {
              data: { country, adult },
              emailRedirectTo: window.location.origin,
            },
          })
      if (result.error) throw result.error
      if (!needsProfile && !codeSent) {
        setEmail(email.trim())
        setCodeSent(true)
      }
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : String(problem))
    } finally {
      setBusy(false)
    }
  }

  async function signOut() {
    const { error: problem } = await client.auth.signOut()
    if (problem) throw problem
    setCode('')
    setCodeSent(false)
    setCountry('')
    setAdult(false)
    setError('')
  }

  if (session === undefined) return <div className="center-state">{t('loading')}</div>
  if (session && !needsProfile) return children(session.user.email ?? '', signOut)
  return (
    <main className="account-page">
      <form className="account-card" onSubmit={submit}>
        <h1>{t(needsProfile ? 'confirmAccess' : 'signIn')}</h1>
        <p>{t(codeSent && !needsProfile ? 'codeSent' : 'signInBody')}</p>
        {!needsProfile && (
          <label className="field">
            <span>{t('email')}</span>
            <input type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} required disabled={busy || codeSent} />
          </label>
        )}
        {!codeSent || needsProfile ? (
          <>
            <label className="field">
              <span>{t('country')}</span>
              <select value={country} onChange={(e) => setCountry(e.target.value)} required disabled={busy}>
                <option value="">{t('chooseCountry')}</option>
                {regions.map((region) => <option key={region} value={region}>{countries.of(region) ?? region}</option>)}
              </select>
            </label>
            <label className="agreement">
              <input type="checkbox" checked={adult} onChange={(e) => setAdult(e.target.checked)} required disabled={busy} />
              {t('adult')}
            </label>
          </>
        ) : (
          <label className="field">
            <span>{t('code')}</span>
            <input type="text" inputMode="numeric" autoComplete="one-time-code" value={code}
              onChange={(e) => setCode(e.target.value)} required disabled={busy} />
          </label>
        )}
        {error && <p className="error-text" role="alert">{error}</p>}
        <button type="submit" className="button primary" disabled={busy}>
          {t(needsProfile ? 'continue' : codeSent ? 'verifyCode' : 'sendSignInEmail')}
        </button>
        {needsProfile ? (
          <button type="button" className="text-button" onClick={() => void signOut().catch((e) => setError(e.message))}>{t('signOut')}</button>
        ) : codeSent && (
          <button type="button" className="text-button" onClick={() => { setCodeSent(false); setCode(''); setError('') }} disabled={busy}>{t('changeEmail')}</button>
        )}
      </form>
    </main>
  )
}
