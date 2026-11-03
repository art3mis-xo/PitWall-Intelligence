// // src/App.tsx
// import { createBrowserRouter, RouterProvider } from 'react-router-dom'
// import Landing from './pages/Landing'
// import Home from './pages/Home'
// import RaceDashboard from './pages/RaceDashboard'
// import Predict from './pages/Predict'

// const router = createBrowserRouter([
//   {
//     path: '/',
//     element: <Landing />,
//   },
//   {
//     path: '/explore',
//     element: <Home />,
//   },
//   {
//     path: '/race/:year/:gp',
//     element: <RaceDashboard />,
//   },
//   {
//     path: '/predict',
//     element: <Predict />,
//   },
// ])

// export default function App() {
//   return (
//     <div className="min-h-screen bg-gray-950 text-white">
//       <RouterProvider router={router} />
//     </div>
//   )
// }

import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Landing       from './pages/Landing'
import Home          from './pages/Home'
import RaceDashboard from './pages/RaceDashboard'
import Predict       from './pages/Predict'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"               element={<Landing />} />
        <Route path="/explore"        element={<Home />} />
        <Route path="/race/:year/:gp" element={<RaceDashboard />} />
        <Route path="/predict"        element={<Predict />} />
      </Routes>
    </BrowserRouter>
  )
}