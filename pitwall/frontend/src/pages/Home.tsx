// src/pages/Home.tsx
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import client from '../api/client'

interface Event {
  round:    number
  name:     string
  country:  string
  date:     string
  format:   string
}

interface Session {
  number: number
  name:   string
}

interface Driver {
  number:       string
  abbreviation: string
  full_name:    string
  team:         string
  team_color:   string
}

export default function Home() {
  const navigate = useNavigate()

  // ── Selector state ─────────────────────────────────────────────
  const currentYear = new Date().getFullYear()
const [years] = useState<number[]>(
  Array.from(
    { length: currentYear - 2017 },
    (_, i) => currentYear - i
  )
)
  const [selectedYear,   setSelectedYear]   = useState<number>(currentYear)
  const [events,         setEvents]         = useState<Event[]>([])
  const [selectedEvent,  setSelectedEvent]  = useState<string>('')
  const [sessions,       setSessions]       = useState<Session[]>([])
  const [selectedSession,setSelectedSession]= useState<string>('')
  const [drivers,        setDrivers]        = useState<Driver[]>([])
  const [selectedDrivers,setSelectedDrivers]= useState<string[]>([])  // abbreviations

  const [loadingEvents,  setLoadingEvents]  = useState(false)
  const [loadingSessions,setLoadingSessions]= useState(false)
  const [loadingDrivers, setLoadingDrivers] = useState(false)
  const [apiError,      setApiError]       = useState<string | null>(null)
  const [eventsRequest,  setEventsRequest]  = useState(0)

  // ── Load events when year changes ──────────────────────────────
  useEffect(() => {
    setSelectedEvent('')
    setSelectedSession('')
    setDrivers([])
    setSelectedDrivers([])
    setApiError(null)
    setLoadingEvents(true)

    client.get(`/selectors/events/${selectedYear}`)
      .then(res => setEvents(res.data.events))
      .catch(err => {
        console.error(err)
        setApiError('Could not load race events. Make sure the API server is running, then retry.')
      })
      .finally(() => setLoadingEvents(false))
  }, [selectedYear, eventsRequest])

  const loadEvents = () => {
    setEventsRequest(request => request + 1)
  }

  // ── Load sessions when event changes ──────────────────────────
  useEffect(() => {
    if (!selectedEvent) return
    setSelectedSession('')
    setDrivers([])
    setSelectedDrivers([])
    setLoadingSessions(true)

    // Find round number from selected event name
    const event = events.find(e => e.name === selectedEvent)
    if (!event) return

    client.get(`/selectors/sessions/${selectedYear}/${event.round}`)
      .then(res => setSessions(res.data.sessions))
      .catch(console.error)
      .finally(() => setLoadingSessions(false))
  }, [selectedEvent])

  // ── Load drivers when session changes ─────────────────────────
  useEffect(() => {
    if (!selectedEvent || !selectedSession) return
    setSelectedDrivers([])
    setLoadingDrivers(true)

    client.get(`/selectors/drivers/${selectedYear}/${selectedEvent}/${selectedSession}`)
      .then(res => {
        setDrivers(res.data.drivers)
        // Select all drivers by default
        setSelectedDrivers(res.data.drivers.map((d: Driver) => d.abbreviation))
      })
      .catch(console.error)
      .finally(() => setLoadingDrivers(false))
  }, [selectedSession])

  const toggleDriver = (abbr: string) => {
    setSelectedDrivers(prev =>
      prev.includes(abbr)
        ? prev.filter(d => d !== abbr)
        : [...prev, abbr]
    )
  }

  const toggleAllDrivers = () => {
    setSelectedDrivers(
      selectedDrivers.length === drivers.length
        ? []
        : drivers.map(d => d.abbreviation)
    )
  }

  const canLoad = selectedYear && selectedEvent && selectedSession
                  && selectedDrivers.length > 0

  const handleLoad = () => {
    if (!canLoad) return
    // Encode drivers as comma-separated query param
    const driversParam = selectedDrivers.join(',')
    navigate(
      `/race/${selectedYear}/${encodeURIComponent(selectedEvent)}` +
      `?session=${encodeURIComponent(selectedSession)}` +
      `&drivers=${driversParam}`
    )
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <nav className="border-b border-gray-800 sticky top-0 z-50 bg-gray-950">
        <div className="max-w-4xl mx-auto px-6 flex items-center h-14">
          <button
            onClick={() => navigate('/')}
            className="text-xs text-gray-500 hover:text-white transition-colors"
          >
            ← Back to home
          </button>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-6 py-12">

        {/* Header */}
        <div className="mb-10">
          <h1 className="text-4xl font-bold tracking-tight mb-2">
            Pit<span className="text-red-500">Wall</span>
          </h1>
          <p className="text-gray-400">F1 Intelligence Platform</p>
        </div>

        {apiError && (
          <div className="bg-red-950 border border-red-800 rounded-xl p-4 mb-6
                          text-red-300 text-sm flex items-center justify-between gap-4">
            <span>{apiError}</span>
            <button
              onClick={loadEvents}
              className="text-red-200 hover:text-white font-medium whitespace-nowrap"
            >
              Retry
            </button>
          </div>
        )}

        {/* Selector card */}
        <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6 mb-6">
          <h2 className="text-sm font-medium text-gray-400 uppercase
                         tracking-wider mb-5">
            Select Session
          </h2>

          <div className="grid grid-cols-3 gap-4 mb-6">

            {/* Year */}
            <div>
              <label className="block text-xs text-gray-500 mb-1.5">Year</label>
              <select
                value={selectedYear}
                onChange={e => setSelectedYear(Number(e.target.value))}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg
                           px-3 py-2.5 text-white text-sm focus:outline-none
                           focus:border-red-500"
              >
                {years.map(y => (
                  <option key={y} value={y}>{y}</option>
                ))}
              </select>
            </div>

            {/* Grand Prix */}
            <div>
              <label className="block text-xs text-gray-500 mb-1.5">
                Grand Prix
              </label>
              {loadingEvents ? (
                <div className="w-full bg-gray-800 border border-gray-700
                                rounded-lg px-3 py-2.5 text-gray-500 text-sm">
                  Loading...
                </div>
              ) : (
                <select
                  value={selectedEvent}
                  onChange={e => setSelectedEvent(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg
                             px-3 py-2.5 text-white text-sm focus:outline-none
                             focus:border-red-500"
                >
                  <option value="">Select GP...</option>
                  {events.map(e => (
                    <option key={e.round} value={e.name}>
                      R{e.round} — {e.name}
                    </option>
                  ))}
                </select>
              )}
            </div>

            {/* Session */}
            <div>
              <label className="block text-xs text-gray-500 mb-1.5">
                Session
              </label>
              {loadingSessions ? (
                <div className="w-full bg-gray-800 border border-gray-700
                                rounded-lg px-3 py-2.5 text-gray-500 text-sm">
                  Loading...
                </div>
              ) : (
                <select
                  value={selectedSession}
                  onChange={e => setSelectedSession(e.target.value)}
                  disabled={!selectedEvent}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg
                             px-3 py-2.5 text-white text-sm focus:outline-none
                             focus:border-red-500 disabled:opacity-40"
                >
                  <option value="">Select session...</option>
                  {sessions.map(s => (
                    <option key={s.number} value={s.name}>
                      {s.name}
                    </option>
                  ))}
                </select>
              )}
            </div>
          </div>

          {/* Driver checkboxes */}
          {drivers.length > 0 && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <label className="text-xs text-gray-500 uppercase tracking-wider">
                  Drivers ({selectedDrivers.length}/{drivers.length} selected)
                </label>
                <button
                  onClick={toggleAllDrivers}
                  className="text-xs text-red-400 hover:text-red-300"
                >
                  {selectedDrivers.length === drivers.length
                    ? 'Deselect all'
                    : 'Select all'}
                </button>
              </div>

              <div className="grid grid-cols-4 gap-2">
                {drivers.map(driver => (
                  <label
                    key={driver.abbreviation}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg
                                border cursor-pointer transition-colors text-sm
                                ${selectedDrivers.includes(driver.abbreviation)
                                  ? 'border-gray-600 bg-gray-800'
                                  : 'border-gray-800 bg-gray-900 opacity-50'
                                }`}
                  >
                    <input
                      type="checkbox"
                      checked={selectedDrivers.includes(driver.abbreviation)}
                      onChange={() => toggleDriver(driver.abbreviation)}
                      className="hidden"
                    />
                    {/* Team colour dot */}
                    <span
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ background: driver.team_color }}
                    />
                    <span className="font-medium">{driver.abbreviation}</span>
                    <span className="text-gray-500 text-xs truncate">
                      {driver.team.split(' ')[0]}
                    </span>
                  </label>
                ))}
              </div>
            </div>
          )}

          {loadingDrivers && (
            <p className="text-gray-500 text-sm mt-3">Loading drivers...</p>
          )}
        </div>

        {/* Load button */}
        <button
          onClick={handleLoad}
          disabled={!canLoad}
          className="w-full py-3 bg-red-600 hover:bg-red-500 disabled:bg-gray-800
                     disabled:text-gray-600 disabled:cursor-not-allowed
                     rounded-xl font-medium transition-colors"
        >
          {canLoad
            ? `Load ${selectedEvent} ${selectedYear} — ${selectedSession}`
            : 'Select year, GP and session to continue'}
        </button>

      </div>
    </div>
  )
}