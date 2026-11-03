import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import csv
import json
import re
import time
from pathlib import Path
import torch
import soundfile as sf
import jiwer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# 1. Configuration