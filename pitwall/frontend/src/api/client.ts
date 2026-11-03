// src/api/client.ts
import axios from 'axios'

const client = axios.create({
  baseURL: '/api',   // Vite proxy forwards this to http://localhost:8000
})

export default client