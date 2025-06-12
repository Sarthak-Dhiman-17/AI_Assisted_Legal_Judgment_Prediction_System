import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()

s = pd.Series(['This is a test.', 'Another test.'])

class DummyCleaner:
    def clean_text(self, text):
        return text.lower()

cleaner = DummyCleaner()

result = s.progress_apply(cleaner.clean_text)
print(result)
