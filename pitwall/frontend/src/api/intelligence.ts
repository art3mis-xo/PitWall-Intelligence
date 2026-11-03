import client from './client'

// ── existing types ─────────────────────────────────────────────────────────

export interface NewsItem {
  source:    string
  headline:  string
  link:      string
  published: string
  summary:   string
}

export interface DriverPrediction {
  driver_id:          string
  driver_full_name:   string
  team_id:            string
  grid_position:      number | null
  predicted_position: number
  predicted_rank:     number
  shap_values:        Record<string, number>
  top_factors:        Array<{
    feature:   string
    value:     number
    direction: 'positive' | 'negative'
  }>
}

export interface PredictionResponse {
  season:        number
  round:         number
  predictions:   DriverPrediction[]
  model_version: string
}

export interface ExplanationFactor {
  feature:    string
  label:      string
  shap_value: number
  direction:  'improves' | 'worsens'
}

export interface ExplanationResponse {
  driver_id:        string
  driver_full_name: string
  team_id:          string
  predicted_rank:   number
  grid_position:    number | null
  shap_values:      Record<string, number>
  top_factors:      ExplanationFactor[]
  narrative:        string | null
}

export interface StandingsEntry {
  driver_id:        string
  driver_full_name: string
  team_id:          string
  points:           number
  position:         number
}

// ── new live types ─────────────────────────────────────────────────────────

export interface NextRace {
  season:    number
  round:     number
  name:      string
  circuit:   string
  country:   string
  race_date: string   // ISO string
  format:    string
}

export interface DriverStanding {
  position:  number
  driver_id: string
  full_name: string
  team_id:   string
  points:    number
  wins:      number
}

export interface ConstructorStanding {
  position: number
  team_id:  string
  points:   number
  wins:     number
}

export interface LiveStandings {
  season:       number
  drivers:      DriverStanding[]
  constructors: ConstructorStanding[]
}

export interface RaceResult {
  position:         string
  classified_pos:   string
  driver_id:        string
  full_name:        string
  team_id:          string
  grid:             number
  laps:             number
  status:           string
  points:           number
  time:             string | null
  fastest_lap:      string | null
  fastest_lap_rank: number | null
}

export interface LastRace {
  season:    number
  round:     number
  race_name: string
  circuit:   string
  date:      string
  results:   RaceResult[]
}

// ── existing API calls ─────────────────────────────────────────────────────

export const getNews = async (limit = 8): Promise<NewsItem[]> => {
  const res = await client.get(`/intelligence/news?limit=${limit}`)
  return res.data.articles
}

export const getPrediction = async (
  season: number,
  round: number
): Promise<PredictionResponse> => {
  const res = await client.get(`/predict/${season}/${round}`)
  return res.data
}

export const getExplanation = async (
  season: number,
  round: number,
  driverId: string
): Promise<ExplanationResponse> => {
  const res = await client.get(`/intelligence/explain/${season}/${round}/${driverId}`)
  return res.data
}

export const getChampionshipStandings = async (
  season: number
): Promise<StandingsEntry[]> => {
  const res = await client.get(`/intelligence/championship/${season}`)
  return res.data.standings
}

export const chatStream = async (
  message: string,
  history: Array<{ role: string; content: string }>,
  onChunk: (chunk: string) => void,
  onDone: () => void
): Promise<void> => {
  const response = await fetch('/api/intelligence/chat', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ message, history, stream: true }),
  })

  if (!response.body) throw new Error('No response body')

  const reader  = response.body.getReader()
  const decoder = new TextDecoder()

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    onChunk(decoder.decode(value))
  }
  onDone()
}

// ── new live API calls ─────────────────────────────────────────────────────

export const getNextRace = async (): Promise<NextRace> => {
  const res = await client.get('/live/next-race')
  return res.data
}

export const getLiveStandings = async (season: number): Promise<LiveStandings> => {
  const res = await client.get(`/live/standings/${season}`)
  return res.data
}

export const getLastRace = async (season: number): Promise<LastRace> => {
  const res = await client.get(`/live/last-race/${season}`)
  return res.data
}