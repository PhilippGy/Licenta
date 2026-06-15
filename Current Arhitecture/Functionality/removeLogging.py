import os
import warnings
import logging

# 1. Suppress standard Python & Tqdm warnings
warnings.filterwarnings("ignore")

# 2. Set environment variables to block Hugging Face reports and bars
# (Must be set before importing transformers/sentence_transformers)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# 3. Force active loggers to stay quiet
logging.getLogger("transformers").setLevel(logging.ERROR)

# 4. Programmatic library silencing safely wrapped in try-except statements
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()  # Transformers uses singular 'bar'
except AttributeError:
    pass

try:
    # Hugging Face Hub uses plural 'bars'
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
except (ImportError, AttributeError):
    pass