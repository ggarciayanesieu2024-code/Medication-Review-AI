
"""Shared configuration for the Medication Review AI project."""

import os
from pathlib import Path

from dotenv import load_dotenv


# Project folders and data files.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PATIENTS_DIR = DATA_DIR / "pacientes"

PRESCRIPTIONS_FILE = PROCESSED_DIR / "prescriptions.json"
FRAGMENTS_FILE = PROCESSED_DIR / "fragmentos_cima.json"
CASCADES_FILE = DATA_DIR / "cascadas.json"

CHROMA_DIR = PROJECT_ROOT / "chroma_db"


# Language model configuration.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TEMPERATURE = 0.0
MAX_TOKENS = 2048


# RAG search configuration.
K_RESULTS = 3
COLLECTION_NAME = "cima_fragments"


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
