from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import requests
from bs4 import BeautifulSoup
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
from datetime import datetime
import logging
import re
from typing import List, Dict, Optional
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configuração inicial
app = FastAPI(title="Assistente Legal API", version="1.0")
nlp = spacy.load("pt_core_news_lg")
logger = logging.getLogger(__name__)

# Configurações
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
URL_BASE = "https://www.cm-aveiro.pt"
PDF_PAGES = ["/pages/311", "/pages/400", "/pages/312"]  # Páginas com PDFs relevantes

# --- Funções de Processamento ---
def melhorar_imagem(img: Image.Image) -> Image.Image:
    """Otimiza imagens para OCR"""
    img = img.convert('L')
    img = img.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(3)

def extrair_texto_pdf(url: str) -> str:
    """Extrai texto de PDFs com fallback para OCR"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        text = ""
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if not page_text:  # Fallback para OCR
                    img = melhorar_imagem(page.to_image().original)
                    page_text = pytesseract.image_to_string(img, lang='por+eng')
                text += f"{page_text}\n"
        return text.strip()
    
    except Exception as e:
        logger.error(f"Erro ao processar PDF {url}: {str(e)}")
        return ""

# --- Funções de Busca ---
def buscar_pdfs() -> List[Dict]:
    """Encontra todos os PDFs nas páginas oficiais"""
    documentos = []
    for page in PDF_PAGES:
        try:
            response = requests.get(f"{URL_BASE}{page}", headers=HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                if link['href'].lower().endswith('.pdf'):
                    doc_url = link['href'] if link['href'].startswith('http') else f"{URL_BASE}{link['href']}"
                    documentos.append({
                        "titulo": link.get_text(strip=True)[:150],
                        "url": doc_url,
                        "origem": page
                    })
        except Exception as e:
            logger.error(f"Erro na página {page}: {str(e)}")
    return documentos

def buscar_resposta(pergunta: str, documentos: List[Dict]) -> Dict:
    """Busca semântica nos textos dos PDFs"""
    vectorizer = TfidfVectorizer()
    textos = [extrair_texto_pdf(doc['url']) for doc in documentos]
    
    try:
        tfidf = vectorizer.fit_transform([pergunta] + textos)
        similarities = cosine_similarity(tfidf[0:1], tfidf[1:])
        best_match_idx = np.argmax(similarities)
        
        if similarities.max() > 0.2:  # Threshold de relevância
            doc = documentos[best_match_idx]
            return {
                "resposta": "Encontrei informações relevantes",
                "documento": doc['titulo'],
                "url": doc['url'],
                "similaridade": float(similarities.max()),
                "trechos": [txt[:200] for txt in textos[best_match_idx].split('\n') if pergunta.lower() in txt.lower()][:3]
            }
    except Exception as e:
        logger.error(f"Erro na busca semântica: {str(e)}")
    
    return {"resposta": "Não encontrei informações relevantes"}

# --- Endpoints ---
@app.get("/")
async def root():
    return {
        "status": "online",
        "endpoints": {
            "/docs": "Lista de documentos",
            "/buscar?q=...": "Busca semântica"
        }
    }

@app.get("/docs")
async def listar_documentos():
    return JSONResponse(content=buscar_pdfs())

@app.get("/buscar")
async def buscar(
    q: str = Query(..., min_length=3, description="Termos de busca"),
    limite: int = Query(5, ge=1, le=10)
):
    documentos = buscar_pdfs()[:limite]
    resultado = buscar_resposta(q, documentos)
    return JSONResponse(content=resultado)

# --- Execução Local ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)