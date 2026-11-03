import { useState, useRef, useEffect } from 'react'
import { chatStream } from '../api/intelligence'

interface Message {
  role:    'user' | 'assistant'
  content: string
}

interface ChatPanelProps {
  raceContext?: string
}

const SUGGESTED = [
  'Who will win the next race?',
  'How does Verstappen perform in wet conditions?',
  'Explain the latest race prediction',
  'What is the optimal pit stop strategy?',
]

export default function ChatPanel({ raceContext }: ChatPanelProps) {
  const [messages,  setMessages]  = useState<Message[]>([])
  const [input,     setInput]     = useState('')
  const [streaming, setStreaming] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async (text: string) => {
    if (!text.trim() || streaming) return

    const userMsg: Message = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setStreaming(true)

    // add empty assistant message — we stream into it
    setMessages(prev => [...prev, { role: 'assistant', content: '' }])

    try {
      await chatStream(
        raceContext ? `${raceContext}\n\nUser question: ${text}` : text,
        messages,
        (chunk) => {
          setMessages(prev => {
            const updated = [...prev]
            const last    = updated[updated.length - 1]
            if (last.role === 'assistant') {
              updated[updated.length - 1] = {
                ...last,
                content: last.content + chunk,
              }
            }
            return updated
          })
        },
        () => setStreaming(false)
      )
    } catch {
      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          role:    'assistant',
          content: 'Something went wrong. Please try again.',
        }
        return updated
      })
      setStreaming(false)
    }
  }

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-2xl border border-gray-800 overflow-hidden">

      {/* header */}
      <div className="px-4 py-3 border-b border-gray-800 flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
        <span className="text-sm font-medium text-gray-300">
          Ask PitWall Intelligence
        </span>
        <span className="text-xs text-gray-600 ml-auto">
          Groq · configured model
        </span>
      </div>

      {/* messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">

        {messages.length === 0 && (
          <div className="space-y-2">
            <p className="text-xs text-gray-600 uppercase tracking-wider mb-3">
              Suggested
            </p>
            {SUGGESTED.map((q, i) => (
              <button
                key={i}
                onClick={() => send(q)}
                className="block w-full text-left text-sm text-gray-400
                           hover:text-white px-3 py-2 rounded-lg
                           hover:bg-gray-800 border border-gray-800
                           hover:border-gray-700 transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`max-w-[85%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
              msg.role === 'user'
                ? 'bg-red-600 text-white'
                : 'bg-gray-800 text-gray-200'
            }`}>
              {msg.content || (
                <span className="inline-flex gap-1 items-center">
                  <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce"
                    style={{ animationDelay: '0ms' }} />
                  <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce"
                    style={{ animationDelay: '150ms' }} />
                  <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce"
                    style={{ animationDelay: '300ms' }} />
                </span>
              )}
            </div>
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      {/* input */}
      <div className="p-3 border-t border-gray-800">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send(input)}
            placeholder="Ask about predictions, strategy, regulations..."
            disabled={streaming}
            className="flex-1 bg-gray-800 border border-gray-700 rounded-lg
                       px-3 py-2 text-sm text-white placeholder-gray-600
                       focus:outline-none focus:border-red-500
                       disabled:opacity-50"
          />
          <button
            onClick={() => send(input)}
            disabled={!input.trim() || streaming}
            className="px-4 py-2 bg-red-600 hover:bg-red-500
                       disabled:bg-gray-800 disabled:text-gray-600
                       rounded-lg text-sm font-medium transition-colors
                       disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </div>

    </div>
  )
}