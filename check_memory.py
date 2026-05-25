# save as check_memory.py and run it
import psutil, os
from app.model import SentimentModel

process = psutil.Process(os.getpid())
before = process.memory_info().rss / 1024 / 1024
print(f"Before model load: {before:.1f} MB")

model = SentimentModel()

after = process.memory_info().rss / 1024 / 1024
print(f"After model load:  {after:.1f} MB")
print(f"Model used:        {after - before:.1f} MB")