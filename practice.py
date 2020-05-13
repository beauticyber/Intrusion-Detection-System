import warnings
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

import pandas as pd
import numpy as np
from utils import get_data, generate_output, guess_human, seed_sequence, get_embeddings, find_closest