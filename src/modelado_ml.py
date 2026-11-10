"""Pipelines y evaluación supervisada del TP."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    RandomizedSearchCV,
    KFold,
)
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy.stats import randint, uniform, loguniform

ESPACIOS_BUSQUEDA = {
    'RandomForest': {
        'regressor__n_estimators': randint(100, 600),
        'regressor__max_depth': randint(3, 25),
        'regressor__min_samples_split': randint(2, 20),
        'regressor__min_samples_leaf': randint(1, 15),
        'regressor__max_features': ['sqrt', 'log2', 0.5, 0.75, 1.0],
        'regressor__max_leaf_nodes': [None, 20, 50, 100, 200],
        'regressor__max_samples': [None, 0.6, 0.8, 1.0],
        'regressor__ccp_alpha': [0.0, 0.0001, 0.001, 0.01],
    },
    'GradientBoosting': {
        'regressor__n_estimators': randint(100, 600),
        'regressor__learning_rate': loguniform(0.01, 0.3),
        'regressor__max_depth': randint(2, 7),
        'regressor__subsample': uniform(0.6, 0.4),
        'regressor__min_samples_split': randint(2, 20),
        'regressor__min_samples_leaf': randint(1, 15),
        'regressor__max_features': [None, 'sqrt', 0.5, 0.75],
        'regressor__max_leaf_nodes': [None, 10, 20, 50],
        'regressor__loss': ['squared_error', 'huber'],
        'regressor__ccp_alpha': [0.0, 0.0001, 0.001],
    },
    'KNN': {
        'regressor__n_neighbors': randint(2, 50),
        'regressor__weights': ['uniform', 'distance'],
        'regressor__p': [1, 2, 3],
        'regressor__leaf_size': randint(15, 60),
    },
}


def crear_pipelines(numeric_features, categorical_features, binary_features, *, semilla):
    """Construye modelos con codificación de zona y escalado para KNN."""
    modelos = {
        "RandomForest": RandomForestRegressor(random_state=semilla),
        "GradientBoosting": GradientBoostingRegressor(random_state=semilla),
        "KNN": KNeighborsRegressor(),
    }
    pipelines = {}
    for nombre, modelo in modelos.items():
        preprocesador = ColumnTransformer([
            ("num", StandardScaler() if nombre == "KNN" else "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("bin", "passthrough", binary_features),
        ])
        pipelines[nombre] = Pipeline([
            ("preprocessor", preprocesador),
            ("regressor", modelo),
        ])
    return pipelines


def _metricas(modelo, X, y):
    pred = modelo.predict(X)
    mse = mean_squared_error(y, pred)
    return {"MAE": mean_absolute_error(y, pred), "MSE": mse,
            "RMSE": np.sqrt(mse), "R2": r2_score(y, pred)}


def evaluar_modelos_base(pipelines, X_train, y_train, X_test, y_test):
    """Ajusta los modelos base y devuelve las métricas de Train y Test."""
    filas = {}
    for nombre, pipeline in pipelines.items():
        modelo = clone(pipeline).fit(X_train, y_train)
        filas[nombre, "Train"] = _metricas(modelo, X_train, y_train)
        filas[nombre, "Test"] = _metricas(modelo, X_test, y_test)
    indice = pd.MultiIndex.from_tuples(filas, names=["Modelo", "Conjunto"])
    return pd.DataFrame(list(filas.values()), index=indice)


def optimizar_modelos(
    pipelines, X_train, y_train, X_test, y_test, *,
    n_iter, n_splits, criterio, semilla, n_jobs=-1,
):
    """Selecciona hiperparámetros en Train y evalúa la selección en Test."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "MSE": "neg_mean_squared_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }
    filas = {}
    busquedas = {}
    for nombre, pipeline in pipelines.items():
        busqueda = RandomizedSearchCV(
            pipeline, ESPACIOS_BUSQUEDA[nombre], n_iter=n_iter, cv=cv,
            scoring=scoring, refit=criterio, n_jobs=n_jobs, random_state=semilla,
        ).fit(X_train, y_train)
        busquedas[nombre] = busqueda
        modelo = busqueda.best_estimator_
        filas[nombre, "Train"] = _metricas(modelo, X_train, y_train)
        filas[nombre, "CV"] = {
            metrica: busqueda.cv_results_[f"mean_test_{metrica}"][busqueda.best_index_]
            * (-1 if scorer.startswith("neg_") else 1)
            for metrica, scorer in scoring.items()
        }
        filas[nombre, "Test"] = _metricas(modelo, X_test, y_test)
    indice = pd.MultiIndex.from_tuples(filas, names=["Modelo", "Conjunto"])
    tabla = pd.DataFrame(list(filas.values()), index=indice)
    return tabla, busquedas


def comparar_resultados(base, optimizado):
    """Compara métricas de prueba y brechas de R² entre Train y Test."""
    base_test = base.xs("Test", level="Conjunto")
    opt_test = optimizado.xs("Test", level="Conjunto")
    return pd.DataFrame({
        "R² Test base": base_test["R2"],
        "R² Test optimizado": opt_test["R2"],
        "RMSE Test base": base_test["RMSE"],
        "RMSE Test optimizado": opt_test["RMSE"],
        "Brecha R² base": base.xs("Train", level="Conjunto")["R2"] - base_test["R2"],
        "Brecha R² optimizado": optimizado.xs("Train", level="Conjunto")["R2"] - opt_test["R2"],
    })


def metricas_por_fold(busquedas):
    """Devuelve las métricas por fold de la configuración seleccionada."""
    filas = []
    for nombre, busqueda in busquedas.items():
        for fold in range(busqueda.n_splits_):
            filas.append({
                "Modelo": nombre,
                "Fold": fold + 1,
                **{
                    metrica: signo * busqueda.cv_results_[f"split{fold}_test_{metrica}"][
                        busqueda.best_index_
                    ]
                    for metrica, signo in {"MAE": -1, "MSE": -1, "RMSE": -1, "R2": 1}.items()
                },
            })
    return pd.DataFrame(filas)
