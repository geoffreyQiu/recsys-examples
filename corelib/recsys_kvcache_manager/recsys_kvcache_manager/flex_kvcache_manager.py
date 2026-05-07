import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from typing import Any, List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

