import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import GridSenseDashboard from './Gridsensedashboard'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <GridSenseDashboard />
  </StrictMode>,
)
