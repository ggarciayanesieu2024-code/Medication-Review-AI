
"""Shared configuration for the Medication Review AI project."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project folders and data files.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
NOMENCLATOR_DIR = RAW_DIR / "nomenclator"
CIMA_RAW_DIR = RAW_DIR / "cima"
PROCESSED_DIR = DATA_DIR / "processed"
PATIENTS_DIR = DATA_DIR / "pacientes"

NOMENCLATOR_ZIP_FILE = NOMENCLATOR_DIR / "prescripcion.zip"
NOMENCLATOR_XML_FILE = NOMENCLATOR_DIR / "Prescripcion.xml"
PRESCRIPTIONS_FILE = PROCESSED_DIR / "prescriptions.json"
FRAGMENTS_FILE = PROCESSED_DIR / "fragmentos_cima.json"
CASCADES_FILE = DATA_DIR / "cascadas.json"

EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DATABASE_DIR = EMBEDDINGS_DIR / "chroma_db"
MODEL_CACHE_DIR = EMBEDDINGS_DIR / "E5_model"


# Language model configuration.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TEMPERATURE = 0.0
MAX_TOKENS = 4096
TOP_P = 1.0
REASONING_EFFORT = "medium"
LLM_TIMEOUT_SECONDS = 60.0
LLM_MAX_RETRIES = 2  # Network/API retries, separate from future JSON validation retries.


# Embedding model configuration: indexing and retrieval must use the same settings.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DEVICE = "cpu"
EMBEDDING_BATCH_SIZE = 4
NORMALIZE_EMBEDDINGS = True
DISTANCE_METRIC = "cosine"


# RAG search configuration.
K_RESULTS = 3
COLLECTION_NAME = "cima_fragments"
MIN_QUOTE_WORDS = 6


# Preliminary phrases to flag unsafe treatment instructions.
FORBIDDEN_PHRASES = (
    "stop taking",
    "discontinue the medication",
    "reduce the dose",
    "increase the dose",
    "start taking",
    "switch to another medicine",
    "deje de tomar",
    "suspenda el medicamento",
    "reduzca la dosis",
    "aumente la dosis",
)
