// src/api/client.ts
import axios from 'axios'

// Locally VITE_API_URL is unset and the Vite proxy forwards /api to http://localhost:8000.
// In production it is the backend's URL, e.g. https://pitwall-api.onrender.com
export const API_BASE = `${import.meta.env.VITE_API_URL ?? ''}/api`

const client = axios.create({
  baseURL: API_BASE,
})

// Prefer the backend's explanation (FastAPI's `detail`) over axios's generic message
export function errorMessage(err: unknown): string {
  if (axios.isAxiosError(err) && typeof err.response?.data?.detail === 'string') {
    return err.response.data.detail
  }
  return err instanceof Error ? err.message : String(err)
}

export default client
