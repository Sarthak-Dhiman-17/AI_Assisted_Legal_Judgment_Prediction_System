from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import joblib

class BaselineModel:
    def __init__(self):
        self.model = make_pipeline(
            TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                sublinear_tf=True
            ),
            SMOTE(random_state=42),
            RandomForestClassifier(
                n_estimators=300,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            )
        )
        
    def train(self, X, y):
        self.model.fit(X, y)
        
    def save(self, path):
        joblib.dump(self.model, path)
        
    @classmethod
    def load(cls, path):
        model = joblib.load(path)
        return model
