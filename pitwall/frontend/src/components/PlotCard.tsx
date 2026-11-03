// src/components/PlotCard.tsx
import { useEffect, useState, useRef } from 'react'
import client from '../api/client'

interface PlotCardProps {
  title: string
  endpoint: string
  params: {
    session: string
    drivers: string
  }
}

export default function PlotCard({ title, endpoint, params }: PlotCardProps) {
  const plotRef = useRef<HTMLDivElement>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [chartHeight, setChartHeight] = useState('500px')
  const [plotData, setPlotData] = useState<any>(null)

  // Stage 1: Fetch data
  useEffect(() => {
    let isMounted = true
    setLoading(true)
    setError(null)

    client.get(endpoint, { params })
      .then(res => {
        if (!isMounted) return
        setPlotData(res.data)
        
        // Update height state if backend provides it
        if (res.data.layout?.height) {
          setChartHeight(`${res.data.layout.height}px`)
        } else {
          setChartHeight('500px')
        }
        setLoading(false)
      })
      .catch(err => {
        if (!isMounted) return
        setError(err.message)
        setLoading(false)
      })

    return () => { isMounted = false }
  }, [endpoint, params.session, params.drivers])

  // Stage 2: Initialize Plotly when data is ready and DOM is rendered
  useEffect(() => {
    if (!loading && plotData && plotRef.current) {
      import('plotly.js-dist-min')
        .then(module => {
          const Plotly = module.default || module
          const target = plotRef.current
          if (!target) return

          Plotly.newPlot(
            target,
            plotData.data,
            {
              ...plotData.layout,
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor:  'rgba(0,0,0,0)',
              font:  { color: '#e5e7eb' },
              xaxis: { ...plotData.layout?.xaxis, gridcolor: '#1f2937', color: '#9ca3af' },
              yaxis: { ...plotData.layout?.yaxis, gridcolor: '#1f2937', color: '#9ca3af' },
              legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: '#9ca3af', size: 11 } },
            },
            { displayModeBar: false, responsive: true }
          )
        })
        .catch(err => {
          console.error('Plotly render error:', err)
        })
    }
  }, [loading, plotData])

  return (
    <div className="bg-gray-900 rounded-xl p-4 border border-gray-800 h-full">
      <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-4">
        {title}
      </h2>
      
      {loading ? (
        <div style={{ height: chartHeight }} className="flex items-center justify-center">
          <div className="text-center">
            <div className="w-6 h-6 border-2 border-red-500 border-t-transparent
                          rounded-full animate-spin mx-auto mb-2" />
            <p className="text-gray-500 text-xs">Loading data...</p>
          </div>
        </div>
      ) : error ? (
        <div style={{ height: chartHeight }} className="flex items-center justify-center text-red-400 text-sm">
          Error: {error}
        </div>
      ) : (
        <div ref={plotRef} style={{ width: '100%', height: chartHeight }} />
      )}
    </div>
  )
}
