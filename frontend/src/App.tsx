import React, { useState, useCallback } from 'react'
import axios from 'axios'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000'

interface UploadResponse {
  pdf_path: string
  page_count: number
  thumbnail: string | null
}

interface SeriesMeta {
  name: string
  axis: 'left' | 'right' | 'none'
  render: 'line' | 'bar'
}

interface TableData {
  page: number
  region: number
  image?: string
  data: Record<string, any>[]
  series_meta?: SeriesMeta[]
  chart_type?: string
  confidence?: 'high' | 'medium' | 'low'
  note?: string
  category?: string
  series_hints?: string[]
  extras_used?: string[]
  error?: string

  // New optional fields coming from backend:
  layout_order?: number
  label?: string

  // Backend flags to indicate full data is present (no truncation):
  truncated?: boolean
  has_more_rows?: boolean
  row_count?: number
  total_rows?: number
  visible_rows?: number
  hidden_rows?: number
  remaining_rows?: number
  more_rows?: number
  preview_count?: number
  ui_has_more_rows?: boolean
}

interface ProcessResponse {
  tables: TableData[]
  debug_raw: any[]
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [pdfPath, setPdfPath] = useState<string>('')
  const [pageCount, setPageCount] = useState<number>(0)
  const [thumbnail, setThumbnail] = useState<string>('')
  const [currentPage, setCurrentPage] = useState<number>(1)
  const [tables, setTables] = useState<TableData[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string>('')
  const [showNotes, setShowNotes] = useState(false)

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0]
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile)
      setError('')
    } else {
      setError('Please select a valid PDF file')
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setIsUploading(true)
    setError('')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post<UploadResponse>(
        `${API_BASE_URL}/upload`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      )

      setPdfPath(response.data.pdf_path)
      setPageCount(response.data.page_count)
      
      if (response.data.thumbnail) {
        setThumbnail(`${API_BASE_URL}/thumbnail/${response.data.thumbnail}?v=${Date.now()}`)
      }
      
      setCurrentPage(1)
      setTables([])
    } catch (err) {
      const errorMessage = axios.isAxiosError(err) 
        ? err.response?.data?.message || err.message
        : 'Upload failed'
      setError(errorMessage)
    } finally {
      setIsUploading(false)
    }
  }

  const handleProcessPage = async () => {
    if (!pdfPath) return

    setIsProcessing(true)
    setError('')

    try {
      const response = await axios.post<ProcessResponse>(
        `${API_BASE_URL}/process_page`,
        {
          pdf_path: pdfPath,
          page_number: currentPage
        }
      )

      setTables(response.data.tables)
      
      // Update thumbnail for current page
      const thumbName = `${pdfPath}_page_${String(currentPage).padStart(3, '0')}_thumb.png`
      setThumbnail(`${API_BASE_URL}/thumbnail/${thumbName}?v=${Date.now()}`)
    } catch (err) {
      const errorMessage = axios.isAxiosError(err)
        ? err.response?.data?.message || err.message
        : 'Processing failed'
      setError(errorMessage)
    } finally {
      setIsProcessing(false)
    }
  }

  const getConfidenceBadgeClass = (confidence?: string) => {
    switch (confidence) {
      case 'high': return 'badge-high'
      case 'medium': return 'badge-medium'
      case 'low': return 'badge-low'
      default: return 'badge-unknown'
    }
  }

  const renderTable = useCallback((table: TableData) => {
    if (table.error) {
      return (
        <div className="region-card error-card" key={`${table.page}-${table.region}`}>
          <h3>Region {table.region}</h3>
          <div className="error-message">{table.error}</div>
          {table.note && <div className="note">{table.note}</div>}
        </div>
      )
    }

    const hasData = !!table.data && table.data.length > 0

    // ---- PREVIEW CONTROL ----
    // Keep as `null` to show ALL rows and NO "more rows" hint.
    // If you ever want a capped preview, set this to a number (e.g., 50).
    const previewLimit: number | null = null

    const columns = hasData
      ? ['X', ...Object.keys(table.data[0]).filter(c => c !== 'X')]
      : []

    const isPreviewing =
      previewLimit !== null &&
      hasData &&
      table.data.length > previewLimit

    const previewData = hasData
      ? (isPreviewing ? table.data.slice(0, previewLimit) : table.data)
      : []

    // Backend-supplied “no truncation” flags (present in your new backend)
    const backendSaysNoMore =
      table.truncated === false &&
      table.has_more_rows === false &&
      (table.remaining_rows ?? 0) === 0 &&
      (table.more_rows ?? 0) === 0

    const showMoreRowsMsg = isPreviewing && !backendSaysNoMore

    return (
      <div className="region-card" key={`${table.page}-${table.region}`}>
        {table.image && (
          <div className="crop-preview">
            <img 
              src={`${API_BASE_URL}/images/enhanced/${table.image}`} 
              alt={`Region ${table.region}`}
            />
          </div>
        )}
        
        <div className="region-info">
          <h3>
            Region {table.region}
            {typeof table.layout_order === 'number' && (
              <span className="chip" style={{ marginLeft: 8 }}>Order {table.layout_order}</span>
            )}
            {table.label && (
              <span className="chip" style={{ marginLeft: 8 }}>{table.label}</span>
            )}
          </h3>
          
          <div className="metadata">
            {table.chart_type && (
              <span className="chip">📊 {table.chart_type}</span>
            )}
            {table.confidence && (
              <span className={`badge ${getConfidenceBadgeClass(table.confidence)}`}>
                {table.confidence}
              </span>
            )}
            {table.category && (
              <span className="chip">{table.category}</span>
            )}
          </div>

          {hasData && (
            <div className="data-preview">
              <table>
                <thead>
                  <tr>
                    {columns.map(col => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewData.map((row, idx) => (
                    <tr key={idx}>
                      {columns.map(col => (
                        <td key={col}>
                          {typeof row[col] === 'number' 
                            ? (Number.isFinite(row[col]) ? row[col].toFixed(2) : row[col])
                            : row[col]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>

              {showMoreRowsMsg && previewLimit !== null && (
                <div className="more-rows">
                  … and {table.data.length - previewLimit} more rows
                </div>
              )}
            </div>
          )}

          {showNotes && table.note && (
            <div className="note">
              <strong>Note:</strong> {table.note}
            </div>
          )}

          {table.note?.includes('CV fallback') && (
            <div className="cv-fallback-hint">
              ⚡ Deterministic extraction applied
            </div>
          )}
        </div>
      </div>
    )
  }, [showNotes])

  return (
    <div className="app">
      <header className="app-header">
        <h1>PDF Chart Data Extractor</h1>
        <p>Extract structured data from charts in PDF documents</p>
      </header>

      <main className="app-main">
        <section className="upload-section">
          <div className="upload-controls">
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileSelect}
              disabled={isUploading || isProcessing}
            />
            <button 
              onClick={handleUpload}
              disabled={!file || isUploading || isProcessing}
              className="btn btn-primary"
            >
              {isUploading ? 'Uploading...' : 'Upload PDF'}
            </button>
          </div>

          {error && (
            <div className="error-banner">
              {error}
            </div>
          )}

          {pdfPath && (
            <div className="pdf-info">
              <p>📄 <strong>PDF loaded:</strong> {pageCount} pages</p>
              
              <div className="page-controls">
                <button
                  onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                  disabled={currentPage <= 1 || isProcessing}
                  className="btn btn-small"
                >
                  Previous
                </button>
                
                <span className="page-indicator">
                  Page {currentPage} of {pageCount}
                </span>
                
                <button
                  onClick={() => setCurrentPage(Math.min(pageCount, currentPage + 1))}
                  disabled={currentPage >= pageCount || isProcessing}
                  className="btn btn-small"
                >
                  Next
                </button>
                
                <button
                  onClick={handleProcessPage}
                  disabled={isProcessing}
                  className="btn btn-primary"
                >
                  {isProcessing ? 'Processing...' : 'Process Page'}
                </button>
              </div>

              {thumbnail && (
                <div className="thumbnail-preview">
                  <img src={thumbnail} alt={`Page ${currentPage}`} />
                </div>
              )}
            </div>
          )}
        </section>

        {tables.length > 0 && (
          <section className="results-section">
            <div className="results-header">
              <h2>Extracted Data</h2>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={showNotes}
                  onChange={(e) => setShowNotes(e.target.checked)}
                />
                Show notes
              </label>
            </div>
            
            <div className="regions-grid">
              {[...tables]
                .sort((a, b) => (a.layout_order ?? 1) - (b.layout_order ?? 1))
                .map(renderTable)}
            </div>
          </section>
        )}

        {isProcessing && (
          <div className="loading-overlay">
            <div className="spinner"></div>
            <p>Extracting chart data...</p>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
