# PDF Chart Data Extractor

A full-stack project that extracts tabular data from chart images embedded in PDFs.  
It combines machine learning with computer vision for high-accuracy results.  

Key features:
- **Flask Backend (Python)** — orchestrates a multi-stage extraction pipeline.  
- **Upstage Document AI** — detects and crops chart regions from PDFs with high precision.  
- **LLM (OpenAI GPT-4o)** — interprets chart crops into structured tabular data.  
- **OpenCV Fallback** — ensures robustness when AI crops fail.  
- **React + TypeScript Frontend** — displays extracted tables directly in the browser using a dynamic React table.  
- **Dockerized Setup** — backend and frontend run in separate containers for clean development and deployment.  

This project demonstrates skills in AI integration, computer vision, cloud APIs, containerization, and building a polished full-stack workflow from document upload to structured data output.
