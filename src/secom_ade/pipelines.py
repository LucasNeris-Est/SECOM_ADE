
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# Importe seu transformador personalizado
from secom_ade.processing.transformers import ColumnDropperByNa

def create_preprocessing_pipeline(nan_threshold=0.4):
    """
    Cria e retorna o pipeline de pré-processamento de dados.
    """
    preprocessor = Pipeline(steps=[
        ('remover_por_na', ColumnDropperByNa(threshold=nan_threshold)),
        ("remover_constantes", VarianceThreshold()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    return preprocessor