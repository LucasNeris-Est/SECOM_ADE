from sklearn.base import BaseEstimator, TransformerMixin

class ColumnDropperByNa(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.columns_to_keep_ = []

    def fit(self, X, y=None):
        # Aprende quais colunas manter APENAS dos dados de treino (X)
        nan_frac = X.isna().mean()
        self.columns_to_keep_ = X.columns[nan_frac < self.threshold].tolist()
        return self

    def transform(self, X, y=None):
        # Aplica a transformação, mantendo apenas as colunas aprendidas
        return X[self.columns_to_keep_]