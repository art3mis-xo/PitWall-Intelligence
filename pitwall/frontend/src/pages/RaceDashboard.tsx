// src/pages/RaceDashboard.tsx
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import PlotCard from '../components/PlotCard'
import GearShiftCard from '../components/GearShiftCard'

export default function RaceDashboard() {
  const navigate = useNavigate()
  const { year, gp }       = useParams()
  const [searchParams]      = useSearchParams()
  const session             = searchParams.get('session') || 'R'
  const drivers             = searchParams.get('drivers') || ''

  if (!year || !gp) return null

  const params = { session, drivers }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
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
            <span className="text-gray-600 font-normal ml-2 text-xs normal-case tracking-normal">
              Intelligence
            </span>
          </button>
        </div>
      </nav>

      <div className="p-6 max-w-7xl mx-auto">
      {/* Header section */}
      <div className="mb-8">
        <div className="flex items-center gap-4">
          <div className="bg-red-600 text-white text-xs font-bold px-2 py-1 rounded">
            {year}
          </div>
          <h1 className="text-3xl font-bold tracking-tight">
            {gp}
          </h1>
        </div>
        <p className="text-gray-400 mt-2 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          {session} Session · Live Telemetry
        </p>
      </div>

      {/* Grid of plots */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Plot 1: Position Changes or Lap Times */}
        <PlotCard 
          title={session === 'Race' || session === 'Sprint' ? 'Position Changes' : 'Lap Times'}
          endpoint={`/telemetry/position-changes/${year}/${gp}`}
          params={params}
        />

        {/* You can add Plot 2, 3, etc. here just by adding more PlotCard components */}
        {/* 
        <PlotCard 
          title="Speed Comparison"
          endpoint={`/telemetry/speed-trace/${year}/${gp}`}
          params={params}
        /> 
        */}
        <PlotCard 
          title="Tyre Strategy"
          endpoint={`/telemetry/tyre-strategy/${year}/${gp}`}
          params={params}
        /> 

        <PlotCard
          title="Speed Traces · Shared Corner Annotations"
          endpoint={`/telemetry/speed-traces/${year}/${gp}`}
          params={params}
        />

        <GearShiftCard
          endpoint={`/telemetry/gear-shifts/${year}/${gp}`}
          params={params}
        />
      </div>
      </div>
    </div>
  )
}
