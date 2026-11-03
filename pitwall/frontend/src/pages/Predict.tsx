import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  getPrediction,
  getExplanation,
  getNextRace,
} from '../api/intelligence'
import type {
  DriverPrediction,
  ExplanationResponse,
} from '../api/intelligence'
import ShapWaterfall from '../components/ShapWaterfall'
import ChatPanel     from '../components/ChatPanel'
import ReactMarkdown from 'react-markdown'
import remarkGfm      from 'remark-gfm'

// Next race is fetched dynamically from the backend

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

export default function Predict() {
  const navigate = useNavigate()

  const [predictions, setPredictions] = useState<DriverPrediction[]>([])
  const [loading,     setLoading]     = useState(true)
  const [error,       setError]       = useState<string | null>(null)

  const [nextRace, setNextRace] = useState<{ season: number; round: number; name: string } | null>(null)

  const [selected,   setSelected]   = useState<DriverPrediction | null>(null)
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null)
  const [loadingExp,  setLoadingExp]  = useState(false)

  const hasQualiData = predictions.some(
  p => p.grid_position !== null && p.grid_position <= 20
)
  useEffect(() => {
    let mounted = true

    async function load() {
      try {
        const nr = await getNextRace()
        if (!mounted) return
        setNextRace(nr)

        const data = await getPrediction(nr.season, nr.round)
        if (!mounted) return
        setPredictions(data.predictions)
        if (data.predictions.length > 0) {
          handleSelect(data.predictions[0], nr)
        }
      } catch (err: any) {
        setError(err?.message || String(err))
      } finally {
        if (mounted) setLoading(false)
      }
    }

    load()
    return () => { mounted = false }
  }, [])

  const handleSelect = async (
    driver: DriverPrediction,
    race = nextRace,
  ) => {
    setSelected(driver)
    setExplanation(null)
    setLoadingExp(true)
    try {
      const season = race?.season ?? 2025
      const round  = race?.round  ?? 24
      const exp = await getExplanation(season, round, driver.driver_id)
      setExplanation(exp)
    } catch {
      // explanation failed silently — SHAP waterfall still shows from prediction data
    } finally {
      setLoadingExp(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">

      {/* ── Nav ── */}
      <nav className="border-b border-gray-800 sticky top-0 z-50 bg-gray-950">
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between h-14">
          <button
            onClick={() => navigate('/')}
            className="text-xs text-gray-500 hover:text-white transition-colors"
          >
            ← Back to home
          </button>
          <button
            onClick={() => navigate('/')}
            className="font-bold tracking-widest text-sm uppercase"
          >
            Pit<span className="text-red-500">Wall</span>
            <span className="text-gray-600 font-normal ml-2 text-xs
                             normal-case tracking-normal">
              Intelligence
            </span>
          </button>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-8">

        {/* ── Header ── */}
        <div className="mb-8">
          <div className="text-xs text-red-500 uppercase tracking-widest
                          mb-2 font-medium">
            AI Race Prediction
          </div>
          <h1 className="text-3xl font-bold tracking-tight">
            {nextRace ? `${nextRace.name} ${nextRace.season}` : 'Upcoming race'}
          </h1>
          <p className="text-gray-500 mt-1 text-sm">
            XGBoost · 0.857 Spearman · 87.1% Top-10 hit rate ·
            Click a driver to see SHAP explanation
          </p>
        </div>
        
        {/* // add this in JSX after the mb-8 header div */}
        {!hasQualiData && predictions.length > 0 && (
          <div className="bg-yellow-950 border border-yellow-800 rounded-xl
                          p-3 mb-6 text-yellow-300 text-xs">
            ⚠️ Qualifying hasn't happened yet — grid positions are estimated from the last race.
            Prediction sharpens automatically after Saturday qualifying.
          </div>
        )}
        {/* ── Error state ── */}
        {error && (
          <div className="bg-red-950 border border-red-800 rounded-xl
                          p-4 mb-6 text-red-300 text-sm">
            Failed to load prediction: {error}
          </div>
        )}

        {/* ── 3 col layout ── */}
        <div className="grid grid-cols-[280px_1fr_340px] gap-6">

          {/* col 1: predicted grid */}
          <div className="bg-gray-900 rounded-2xl border border-gray-800 overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-800">
              <div className="text-xs text-gray-500 uppercase tracking-wider">
                Predicted top 10
              </div>
            </div>

            {loading ? (
              <div className="p-4 space-y-2">
                {[...Array(10)].map((_, i) => (
                  <div key={i}
                    className="h-12 bg-gray-800 rounded-lg animate-pulse" />
                ))}
              </div>
            ) : (
              <div className="overflow-y-auto">
                {predictions.slice(0, 10).map(driver => (
                  <button
                    key={driver.driver_id}
                    onClick={() => handleSelect(driver)}
                    className={`w-full flex items-center gap-3 px-4 py-3
                               border-b border-gray-800 last:border-0
                               transition-colors text-left
                               ${selected?.driver_id === driver.driver_id
                                 ? 'bg-gray-800'
                                 : 'hover:bg-gray-800/50'
                               }`}
                  >
                    <div className={`text-sm font-medium w-6 text-right flex-shrink-0 ${
                      driver.predicted_rank <= 3 ? 'text-red-400' : 'text-gray-600'
                    }`}>
                      P{driver.predicted_rank}
                    </div>
                    <div
                      className="w-0.5 h-8 rounded-full flex-shrink-0"
                      style={{ backgroundColor: getTeamColor(driver.team_id) }}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium truncate">
                        {driver.driver_full_name}
                      </div>
                      <div className="text-xs text-gray-500 truncate">
                        {driver.team_id}
                      </div>
                    </div>
                    <div className="text-xs text-gray-600 flex-shrink-0">
                      {driver.grid_position ? `P${driver.grid_position}` : '—'}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* col 2: SHAP explanation */}
          <div className="bg-gray-900 rounded-2xl border border-gray-800 p-5">
            {!selected ? (
              <div className="text-gray-600 text-sm">
                Select a driver to see explanation
              </div>
            ) : (
              <>
                {/* driver header */}
                <div className="flex items-center gap-3 mb-5 pb-5
                                border-b border-gray-800">
                  <div
                    className="w-1 h-10 rounded-full flex-shrink-0"
                    style={{ backgroundColor: getTeamColor(selected.team_id) }}
                  />
                  <div>
                    <div className="text-lg font-bold">
                      {selected.driver_full_name}
                    </div>
                    <div className="text-xs text-gray-500">{selected.team_id}</div>
                  </div>
                  <div className="ml-auto text-right">
                    <div className="text-2xl font-bold text-red-400">
                      P{selected.predicted_rank}
                    </div>
                    <div className="text-xs text-gray-600">predicted</div>
                  </div>
                  {selected.grid_position && (
                    <div className="text-right">
                      <div className="text-lg font-semibold text-gray-400">
                        P{selected.grid_position}
                      </div>
                      <div className="text-xs text-gray-600">grid</div>
                    </div>
                  )}
                </div>

                {/* narrative */}
                {loadingExp ? (
                  <div className="space-y-2 mb-5">
                    <div className="h-4 bg-gray-800 rounded animate-pulse w-full" />
                    <div className="h-4 bg-gray-800 rounded animate-pulse w-5/6" />
                    <div className="h-4 bg-gray-800 rounded animate-pulse w-4/6" />
                  </div>
                ) : explanation?.narrative ? (
                  <div className="bg-gray-800 rounded-xl p-4 mb-5 text-sm
                                  text-gray-300 leading-relaxed space-y-3
                                  [&_strong]:text-white [&_strong]:font-semibold
                                  [&_table]:w-full [&_table]:text-xs
                                  [&_th]:text-left [&_th]:text-gray-500 [&_th]:font-medium
                                  [&_th]:border-b [&_th]:border-gray-700 [&_th]:pb-2
                                  [&_td]:border-b [&_td]:border-gray-700 [&_td]:py-2
                                  [&_td]:pr-3 [&_code]:text-red-300 [&_code]:text-xs">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {explanation.narrative}
                    </ReactMarkdown>
                  </div>
                ) : null}

                {/* SHAP waterfall */}
                {explanation ? (
                  <ShapWaterfall factors={explanation.top_factors} />
                ) : !loadingExp ? (
                  // fallback: use top_factors from prediction if explanation failed
                  <ShapWaterfall
                    factors={selected.top_factors.map(f => ({
                      feature:    f.feature,
                      label:      f.feature.replace(/_/g, ' '),
                      shap_value: f.value,
                      direction:  f.direction === 'positive' ? 'improves' : 'worsens',
                    }))}
                  />
                ) : null}
              </>
            )}
          </div>

          {/* col 3: chat */}
          <div className="h-[640px]">
            <ChatPanel
              raceContext={nextRace
                ? `The active prediction is for the ${nextRace.name}, round ${nextRace.round} of ${nextRace.season}. `
                  + `The model's current predicted top three are: ${predictions.slice(0, 3)
                    .map(driver => `${driver.driver_full_name} (P${driver.predicted_rank}, grid P${driver.grid_position ?? 'unknown'})`)
                    .join('; ')}. Use the prediction data, not invented qualifying results.`
                : undefined}
            />
          </div>

        </div>
      </div>
    </div>
  )
}