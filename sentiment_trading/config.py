from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Medallion-pattern data tiers
DATA_DIR = BASE_DIR / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

# Candidate sentiment transformers (paper Section III-C)
CANDIDATE_MODELS = {
    "deberta-finance": "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis",
    "finbert": "ProsusAI/finbert",
    "pmatorras": "pmatorras/BERT-finance",
}

# FinancialPhraseBank (paper Section III-B): 4,840 sentences = sentences_50agree
DATASET_NAME = "takala/financial_phrasebank"
DATASET_CONFIG = "sentences_50agree"

# Stratified splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Fine-tuning hyperparameters (paper Section III-C)
MAX_SEQ_LEN = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_EPOCHS = 4
EARLY_STOP_PATIENCE = 1

# Temporal decay (paper Section III-G, refs [13][14])
TEMPORAL_DECAY_TAU_MINUTES = 90.0

# Latency budget for lexicon fallback trigger
LATENCY_BUDGET_SECONDS = 2.0
