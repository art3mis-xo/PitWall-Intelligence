import { useEffect, useRef, useState } from 'react'
import client from '../api/client'

interface GearShiftCardProps {
  endpoint: string
  params: { session: string; drivers: string }
}

interface PlotFigure {
  data: any[]
  layout: any
  meta?: { drivers?: string[] }
}

export default function GearShiftCard({ endpoint, params }: GearShiftCardProps) {
  const plotRef = useRef<HTMLDivElement>(null)
  const plotlyRef = useRef<any>(null)
  const [figure, setFigure] = useState<PlotFigure | null>(null)
  const [selectedDriver, setSelectedDriver] = useState('')
  const [compareDriver, setCompareDriver] = useState('')
  const [compareMode, setCompareMode] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let mounted = true
    setLoading(true)
    setError(null)

    client.get(endpoint, { params })
      .then(response => {
        if (!mounted) return
        const nextFigure = response.data as PlotFigure
        const drivers = nextFigure.meta?.drivers ?? []
        setFigure(nextFigure)
        setSelectedDriver(drivers[0] ?? '')
        setCompareDriver(drivers[1] ?? drivers[0] ?? '')
      })
      .catch(err => mounted && setError(err.message))
      .finally(() => mounted && setLoading(false))

    return () => { mounted = false }
  }, [endpoint, params.session, params.drivers])

  useEffect(() => {
    if (!figure || !plotRef.current) return
    const target = plotRef.current

    import('plotly.js-dist-min').then(module => {
      const Plotly = module.default || module
      plotlyRef.current = Plotly
      Plotly.newPlot(
        target,
        figure.data,
        {
          ...figure.layout,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { color: '#e5e7eb' },
          xaxis: { ...figure.layout?.xaxis, gridcolor: '#1f2937' },
          yaxis: { ...figure.layout?.yaxis, gridcolor: '#1f2937' },
        },
        { displayModeBar: false, responsive: true },
      )
    })
  }, [figure])

  useEffect(() => {
    if (!figure || !plotRef.current || !plotlyRef.current) return
    const visibleDrivers = new Set(
      compareMode ? [selectedDriver, compareDriver] : [selectedDriver],
    )
    const visibility = figure.data.map(trace => visibleDrivers.has(trace.meta?.driver))
    plotlyRef.current.restyle(plotRef.current, { visible: visibility })
  }, [figure, selectedDriver, compareDriver, compareMode])

  const drivers = figure?.meta?.drivers ?? []

  return (
    <div className="bg-gray-900 rounded-xl p-4 border border-gray-800 h-full lg:col-span-2">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wider">
          Gear Shifts On Track
        </h2>
        {!loading && !error && (
          <div className="flex items-center gap-3 text-xs">
            <label className="text-gray-400">
              Driver
              <select
                value={selectedDriver}
                onChange={event => setSelectedDriver(event.target.value)}
                className="ml-2 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white"
              >
                {drivers.map(driver => <option key={driver}>{driver}</option>)}
              </select>
            </label>
            <label className="flex items-center gap-2 text-gray-400">
              <input
                type="checkbox"
                checked={compareMode}
                onChange={event => setCompareMode(event.target.checked)}
              />
              Compare
            </label>
            {compareMode && (
              <select
                value={compareDriver}
                onChange={event => setCompareDriver(event.target.value)}
                className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white"
              >
                {drivers.filter(driver => driver !== selectedDriver).map(driver => (
                  <option key={driver}>{driver}</option>
                ))}
              </select>
            )}
          </div>
        )}
      </div>

      {loading ? (
        <div className="h-[500px] flex items-center justify-center text-gray-500 text-xs">
          Loading telemetry...
        </div>
      ) : error ? (
        <div className="h-[500px] flex items-center justify-center text-red-400 text-sm">
          Error: {error}
        </div>
      ) : (
        <div ref={plotRef} className="w-full h-[500px]" />
      )}
    </div>
  )
}