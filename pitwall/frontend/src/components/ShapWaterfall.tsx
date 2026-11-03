import { useState } from 'react'
import type { ExplanationFactor } from '../api/intelligence'

interface Props {
  factors: ExplanationFactor[]
}

export default function ShapWaterfall({ factors }: Props) {
  const [showTooltip, setShowTooltip] = useState(false)

  if (!factors?.length) return null

  const maxAbs = Math.max(...factors.map(f => Math.abs(f.shap_value)), 0.01)

  return (
    <div className="space-y-3">

      {/* heading + tooltip */}
      <div className="flex items-center gap-2 mb-4">
        <div className="text-xs text-gray-500 uppercase tracking-wider">
          Feature contributions
        </div>
        <div className="relative">
          <button
            onMouseEnter={() => setShowTooltip(true)}
            onMouseLeave={() => setShowTooltip(false)}
            className="w-4 h-4 rounded-full bg-gray-700 hover:bg-gray-600
                       text-gray-400 text-xs flex items-center justify-center
                       transition-colors cursor-help"
          >
            i
          </button>
          {showTooltip && (
            <div className="absolute left-6 top-0 z-50 w-64 bg-gray-800
                            border border-gray-700 rounded-xl p-3 shadow-xl">
              <div className="text-xs text-gray-200 font-medium mb-1">
                What are feature contributions?
              </div>
              <div className="text-xs text-gray-400 leading-relaxed">
                These bars show which factors most influenced the model's
                prediction for this driver. Each bar represents one input
                feature — the longer the bar, the bigger the impact.
                <br /><br />
                <span className="text-green-400">Green ↑</span> means the
                factor improves their predicted finishing position (e.g.
                starting from pole pushes the prediction toward P1).
                <br /><br />
                <span className="text-red-400">Red ↓</span> means the factor
                worsens it (e.g. a grid penalty drops them back).
                <br /><br />
                The numbers are SHAP values — a standard ML technique for
                explaining model predictions.
              </div>
            </div>
          )}
        </div>
      </div>

      {/* bars */}
      {factors.map((factor, i) => {
        const improves = factor.direction === 'improves'
        const pct      = Math.abs(factor.shap_value) / maxAbs * 100

        return (
          <div key={i} className="flex items-center gap-3">
            <div className="w-44 text-xs text-gray-400 text-right truncate flex-shrink-0">
              {factor.label}
            </div>
            <div className="flex-1 flex items-center gap-2">
              <div className="flex-1 h-5 bg-gray-800 rounded overflow-hidden">
                <div
                  className={`h-full rounded transition-all duration-500 ${
                    improves ? 'bg-green-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className={`text-xs font-mono w-16 text-right flex-shrink-0 ${
                improves ? 'text-green-400' : 'text-red-400'
              }`}>
                {improves ? '↑' : '↓'} {Math.abs(factor.shap_value).toFixed(3)}
              </div>
            </div>
          </div>
        )
      })}

      <div className="text-xs text-gray-600 mt-4 pt-3 border-t border-gray-800">
        Green ↑ = improves predicted finish · Red ↓ = worsens predicted finish
      </div>
    </div>
  )
}