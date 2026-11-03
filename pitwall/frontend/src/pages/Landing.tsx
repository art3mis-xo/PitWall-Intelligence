import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  getNews, getNextRace, getLiveStandings, getLastRace,
  type NewsItem, type NextRace, type DriverStanding,
  type ConstructorStanding, type RaceResult,
} from '../api/intelligence'

const TEAM_COLORS: Record<string, string> = {
  'Red Bull Racing': '#3671C6',
  'McLaren':         '#FF8000',
  'Ferrari':         '#E8002D',
  'Mercedes':        '#00D2BE',
  'Aston Martin':    '#229971',
  'Alpine':          '#FF87BC',
  'Williams':        '#64C4FF',
  'RB':              '#6692FF',
  'Kick Sauber':     '#52E252',
  'Haas F1 Team':    '#B6BABD',
}

function getTeamColor(teamId: string): string {
  for (const [key, color] of Object.entries(TEAM_COLORS)) {
    if (teamId.includes(key) || key.includes(teamId)) return color
  }
  return '#888888'
}

function useCountdown(targetTime: number | null) {
  const [time, setTime] = useState({ d: 0, h: 0, m: 0, s: 0 })
  useEffect(() => {
    if (targetTime === null) return
    const tick = () => {
      const diff = targetTime - Date.now()
      if (diff <= 0) return
      setTime({
        d: Math.floor(diff / 86400000),
        h: Math.floor((diff % 86400000) / 3600000),
        m: Math.floor((diff % 3600000) / 60000),
        s: Math.floor((diff % 60000) / 1000),
      })
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [targetTime])
  return time
}

function StatusBadge({ status }: { status: string }) {
  const isDNF = status.includes('DNF') || status.includes('Accident')
    || status.includes('Retired') || status.includes('Engine')
    || status.includes('Mechanical') || status.includes('Collision')
  const isFinished = status === 'Finished'
  if (isFinished) return null
  if (isDNF) return (
    <span className="text-xs bg-red-950 text-red-400 border border-red-900
                     px-1.5 py-0.5 rounded font-medium">DNF</span>
  )
  return <span className="text-xs text-gray-500">{status}</span>
}

export default function Landing() {
  const navigate = useNavigate()   // ← MUST be inside the component

  const [nextRace,        setNextRace]        = useState<NextRace | null>(null)
  const [news,            setNews]            = useState<NewsItem[]>([])
  const [driverStandings, setDriverStandings] = useState<DriverStanding[]>([])
  const [constStandings,  setConstStandings]  = useState<ConstructorStanding[]>([])
  const [lastRace,        setLastRace]        = useState<{ name: string; results: RaceResult[] } | null>(null)
  const [standingsTab,    setStandingsTab]    = useState<'drivers' | 'constructors' | 'last'>('drivers')
  const [loadingNews,     setLoadingNews]     = useState(true)
  const [loadingLive,     setLoadingLive]     = useState(true)
  const [currentSeason,   setCurrentSeason]   = useState(new Date().getFullYear())

  const raceTime  = nextRace ? new Date(nextRace.race_date).getTime() : null
  const countdown = useCountdown(raceTime)
  const pad       = (n: number) => String(n).padStart(2, '0')

  useEffect(() => {
    getNews(6).then(setNews).catch(console.error).finally(() => setLoadingNews(false))

    getNextRace()
      .then(r => { setNextRace(r); setCurrentSeason(r.season) })
      .catch(console.error)

    const season = new Date().getFullYear()
    getLiveStandings(season)
      .then(data => {
        setDriverStandings(data.drivers.slice(0, 5))
        setConstStandings(data.constructors.slice(0, 5))
      })
      .catch(console.error)

    getLastRace(season)
      .then(data => setLastRace({ name: data.race_name, results: data.results }))
      .catch(console.error)
      .finally(() => setLoadingLive(false))
  }, [])

  const formatRaceDate = (iso: string) => {
    try {
      return new Date(iso).toLocaleDateString('en-GB', {
        weekday: 'short', day: 'numeric', month: 'short',
        year: 'numeric', hour: '2-digit', minute: '2-digit',
        timeZoneName: 'short',
      })
    } catch { return iso }
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">

      {/* ── Nav ── */}
      <nav className="border-b border-gray-800 sticky top-0 z-50 bg-gray-950">
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between h-14">
          <div className="font-bold tracking-widest text-sm uppercase">
            Pit<span className="text-red-500">Wall</span>
            <span className="text-gray-600 font-normal ml-2 text-xs normal-case tracking-normal">
              Intelligence
            </span>
          </div>
          <div className="flex items-center gap-6">
            <button onClick={() => navigate('/predict')}
              className="text-xs text-gray-400 hover:text-white transition-colors">
              Race predictor
            </button>
            <button onClick={() => navigate('/explore')}
              className="text-xs text-gray-400 hover:text-white transition-colors">
              Past Races
            </button>
            <button onClick={() => navigate('/predict')}
              className="bg-red-600 hover:bg-red-500 text-white text-xs
                         font-medium px-4 py-2 rounded-lg transition-colors">
              Predict next race →
            </button>
          </div>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="border-b border-gray-800 bg-gray-900">
        <div className="max-w-7xl mx-auto px-6 py-10
                        grid grid-cols-[1fr_300px] gap-12 items-end">
          <div>
            <div className="text-xs text-red-500 uppercase tracking-widest mb-4 font-medium">
              {currentSeason} Season
              {nextRace ? ` · Round ${nextRace.round}` : ''}
            </div>
            <h1 className="text-5xl font-bold tracking-tight leading-tight mb-4">
              Race intelligence,<br />
              <span className="text-gray-400 font-light">before the lights go out.</span>
            </h1>
            <p className="text-gray-400 text-base max-w-lg leading-relaxed">
              AI-powered predictions, SHAP-grounded explanations, and live
              championship data — powered by XGBoost and Llama 3.1.
            </p>
            <div className="flex gap-6 mt-6">
              {[
                ['0.857', 'Spearman · 2025 test'],
                ['87.1%', 'Top-10 hit rate'],
                ['3.6k',  'Training rows'],
              ].map(([val, label]) => (
                <div key={label}>
                  <div className={`text-2xl font-bold ${
                    label.includes('Training') ? 'text-white' : 'text-green-400'
                  }`}>{val}</div>
                  <div className="text-xs text-gray-600 mt-1">{label}</div>
                </div>
              ))}
            </div>
          </div>

          {/* countdown */}
          <div className="bg-gray-800 rounded-2xl border border-gray-700 p-5">
            <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">
              Next race
            </div>
            {nextRace ? (
              <>
                <div className="text-base font-semibold mb-0.5">{nextRace.name}</div>
                <div className="text-xs text-gray-500 mb-5">
                  {nextRace.circuit} · {formatRaceDate(nextRace.race_date)}
                </div>
              </>
            ) : (
              <div className="h-8 bg-gray-700 rounded animate-pulse mb-5" />
            )}
            <div className="grid grid-cols-4 gap-2 mb-5">
              {[
                [pad(countdown.d), 'days'],
                [pad(countdown.h), 'hrs'],
                [pad(countdown.m), 'min'],
                [pad(countdown.s), 'sec'],
              ].map(([val, unit]) => (
                <div key={unit} className="text-center bg-gray-900 rounded-xl py-3">
                  <div className="text-2xl font-bold font-mono tabular-nums">{val}</div>
                  <div className="text-xs text-gray-600 mt-1 uppercase tracking-wider">{unit}</div>
                </div>
              ))}
            </div>
            <button
              onClick={() => navigate('/predict')}
              className="w-full py-2.5 bg-red-600 hover:bg-red-500
                         rounded-xl text-sm font-medium transition-colors">
              Get race prediction →
            </button>
          </div>
        </div>
      </section>

      {/* ── 3 col ── */}
      <section className="max-w-7xl mx-auto px-6 py-8
                          grid grid-cols-[1fr_1fr_260px] gap-6">

        {/* standings col — tabbed */}
        <div>
          <div className="flex gap-1 mb-4 pb-3 border-b border-gray-800">
            {([
              ['drivers',      'Drivers'],
              ['constructors', 'Constructors'],
              ['last',         'Last Race'],
            ] as const).map(([key, label]) => (
              <button key={key} onClick={() => setStandingsTab(key)}
                className={`text-xs px-3 py-1.5 rounded-lg transition-colors ${
                  standingsTab === key
                    ? 'bg-gray-800 text-white'
                    : 'text-gray-500 hover:text-gray-300'
                }`}>
                {label}
              </button>
            ))}
          </div>

          {loadingLive ? (
            <div className="space-y-3">
              {[1,2,3,4,5].map(i => (
                <div key={i} className="h-12 bg-gray-800 rounded-lg animate-pulse" />
              ))}
            </div>
          ) : (
            <>
              {standingsTab === 'drivers' && (
                <div className="space-y-1">
                  {driverStandings.map((s, i) => (
                    <div key={s.driver_id}
                      className="flex items-center gap-3 py-3 border-b border-gray-800 last:border-0">
                      <div className={`text-sm font-medium w-5 text-right ${
                        i < 3 ? 'text-red-400' : 'text-gray-600'
                      }`}>{s.position}</div>
                      <div className="w-0.5 h-8 rounded-full flex-shrink-0"
                        style={{ backgroundColor: getTeamColor(s.team_id) }} />
                      <div className="flex-1">
                        <div className="text-sm font-medium">{s.full_name}</div>
                        <div className="text-xs text-gray-500">{s.team_id}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-base font-semibold tabular-nums">{s.points}</div>
                        <div className="text-xs text-gray-600">pts</div>
                      </div>
                    </div>
                  ))}
                  <button onClick={() => setStandingsTab('constructors')}
                    className="text-xs text-gray-600 hover:text-gray-400 mt-2
                               transition-colors w-full text-center pt-2">
                    View constructors →
                  </button>
                </div>
              )}

              {standingsTab === 'constructors' && (
                <div className="space-y-1">
                  {constStandings.map((s, i) => (
                    <div key={s.team_id}
                      className="flex items-center gap-3 py-3 border-b border-gray-800 last:border-0">
                      <div className={`text-sm font-medium w-5 text-right ${
                        i < 3 ? 'text-red-400' : 'text-gray-600'
                      }`}>{s.position}</div>
                      <div className="w-0.5 h-8 rounded-full flex-shrink-0"
                        style={{ backgroundColor: getTeamColor(s.team_id) }} />
                      <div className="flex-1">
                        <div className="text-sm font-medium">{s.team_id}</div>
                        <div className="text-xs text-gray-500">{s.wins} wins</div>
                      </div>
                      <div className="text-right">
                        <div className="text-base font-semibold tabular-nums">{s.points}</div>
                        <div className="text-xs text-gray-600">pts</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {standingsTab === 'last' && lastRace && (
                <div>
                  <div className="text-xs text-gray-500 mb-3">{lastRace.name}</div>
                  <div className="space-y-1 max-h-80 overflow-y-auto">
                    {lastRace.results.map((r, i) => {
                      const isNC = r.classified_pos === 'NC' || r.classified_pos === 'R'
                      return (
                        <div key={i}
                          className="flex items-center gap-3 py-2 border-b border-gray-800 last:border-0">
                          <div className={`text-sm font-medium w-6 text-right flex-shrink-0 ${
                            i < 3 ? 'text-red-400' : isNC ? 'text-gray-700' : 'text-gray-600'
                          }`}>
                            {isNC ? 'NC' : `P${r.classified_pos}`}
                          </div>
                          <div className="w-0.5 h-6 rounded-full flex-shrink-0"
                            style={{ backgroundColor: getTeamColor(r.team_id) }} />
                          <div className="flex-1 min-w-0">
                            <div className={`text-sm font-medium truncate ${isNC ? 'text-gray-600' : ''}`}>
                              {r.full_name}
                            </div>
                            <div className="text-xs text-gray-600">{r.team_id}</div>
                          </div>
                          <div className="flex-shrink-0">
                            <StatusBadge status={r.status} />
                            {r.time && !isNC && (
                              <span className="text-xs text-gray-500">{r.time}</span>
                            )}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* prediction CTA */}
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wider
                          mb-4 pb-3 border-b border-gray-800">
            AI race prediction
          </div>
          <div className="bg-gray-900 rounded-2xl border border-gray-800 p-5 mb-4">
            <div className="text-sm text-gray-400 mb-3">Next prediction</div>
            <div className="text-lg font-bold mb-1">{nextRace?.name ?? 'Loading...'}</div>
            <div className="text-xs text-gray-500 mb-4">
              Post-qualifying · Available after Saturday
            </div>
            <button onClick={() => navigate('/predict')}
              className="w-full py-2.5 bg-gray-800 hover:bg-gray-700
                         border border-gray-700 rounded-xl text-sm
                         font-medium transition-colors mb-3">
              View prediction →
            </button>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gray-800 rounded-xl p-3 text-center">
                <div className="text-lg font-bold text-green-400">0.857</div>
                <div className="text-xs text-gray-600">Spearman</div>
              </div>
              <div className="bg-gray-800 rounded-xl p-3 text-center">
                <div className="text-lg font-bold text-green-400">87%</div>
                <div className="text-xs text-gray-600">Top-10 hit</div>
              </div>
            </div>
          </div>
          <button onClick={() => navigate('/explore')}
            className="w-full py-2.5 bg-gray-900 hover:bg-gray-800
                       border border-gray-800 rounded-xl text-sm
                       font-medium text-gray-300 transition-colors">
            Browse Past Races →
          </button>
        </div>

        {/* news */}
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wider
                          mb-4 pb-3 border-b border-gray-800">
            Latest news
          </div>
          {loadingNews ? (
            <div className="space-y-4">
              {[1,2,3,4].map(i => (
                <div key={i} className="animate-pulse space-y-2">
                  <div className="h-2 bg-gray-800 rounded w-16" />
                  <div className="h-3 bg-gray-800 rounded w-full" />
                  <div className="h-3 bg-gray-800 rounded w-3/4" />
                </div>
              ))}
            </div>
          ) : news.length === 0 ? (
            <div className="text-gray-600 text-sm">No news available.</div>
          ) : (
            <div className="space-y-1">
              {news.map((item, i) => (
                <a key={i} href={item.link} target="_blank" rel="noopener noreferrer"
                  className="block py-3 border-b border-gray-800 last:border-0
                             hover:bg-gray-900 -mx-2 px-2 rounded-lg transition-colors">
                  <div className="text-xs text-red-500 font-medium uppercase tracking-wider mb-1">
                    {item.source}
                  </div>
                  <div className="text-sm font-medium leading-snug text-gray-200 mb-1">
                    {item.headline}
                  </div>
                  <div className="text-xs text-gray-600">{item.published.slice(0, 16)}</div>
                </a>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="border-t border-gray-800 bg-gray-900 mt-4">
        <div className="max-w-7xl mx-auto px-6 py-6 grid grid-cols-4 gap-6">
          {[
            ['XGBoost',       'Prediction model'],
            ['Groq model',    'Agent backbone'],
            ['LangGraph',     'Orchestration'],
            ['Groq',          'Inference engine'],
          ].map(([val, label]) => (
            <div key={label}>
              <div className="text-base font-semibold">{val}</div>
              <div className="text-xs text-gray-600 mt-0.5">{label}</div>
            </div>
          ))}
        </div>
      </footer>
    </div>
  )
}