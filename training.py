# Álvaro Lúcio Almeida Ribeiro
# Engenharia de Software - 163
# C214 L1 - Engenharia de Software

from abc import ABC, abstractmethod
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Interface de Modelo
class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass


# Implementação com Random Forest
class RandomForestModel(BaseModel):
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


# Implementação com Regressão Logística
class LogisticRegressionModel(BaseModel):
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


# Classe de Pipeline
class MLTrainingPipeline:
    def __init__(self, model: BaseModel):
        self.model = model
        self.scaler = StandardScaler()

    def run_pipeline(self, X, y):
        # Dividir os dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Pré-processamento
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Treinamento
        self.model.train(X_train, y_train)

        # Predição
        predictions = self.model.predict(X_test)

        # Avaliação
        accuracy = accuracy_score(y_test, predictions)
        print(f"Acurácia: {accuracy:.2f}")


X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Usando a injeção de dependência para treinar com RandomForest
random_forest_pipeline = MLTrainingPipeline(RandomForestModel())
random_forest_pipeline.run_pipeline(X, y)

# Usando a injeção de dependência para treinar com Logistic Regression
logistic_regression_pipeline = MLTrainingPipeline(LogisticRegressionModel())
logistic_regression_pipeline.run_pipeline(X, y)
