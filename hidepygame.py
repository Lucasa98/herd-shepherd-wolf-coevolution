import os
import warnings

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
