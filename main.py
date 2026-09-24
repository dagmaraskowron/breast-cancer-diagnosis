from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    recall_score
)


images_dir = Path("images")
images_dir.mkdir(exist_ok=True)


feature_names_pl = {
    "mean radius": "Średni promień",
    "mean texture": "Średnia tekstura",
    "mean perimeter": "Średni obwód",
    "mean area": "Średnia powierzchnia",
    "mean smoothness": "Średnia gładkość",
    "mean compactness": "Średnia zwartość",
    "mean concavity": "Średnia wklęsłość",
    "mean concave points": "Średnia liczba punktów wklęsłych",
    "mean symmetry": "Średnia symetria",
    "mean fractal dimension": "Średni wymiar fraktalny",
    "radius error": "Błąd promienia",
    "texture error": "Błąd tekstury",
    "perimeter error": "Błąd obwodu",
    "area error": "Błąd powierzchni",
    "smoothness error": "Błąd gładkości",
    "compactness error": "Błąd zwartości",
    "concavity error": "Błąd wklęsłości",
    "concave points error": "Błąd punktów wklęsłych",
    "symmetry error": "Błąd symetrii",
    "fractal dimension error": "Błąd wymiaru fraktalnego",
    "worst radius": "Największy promień",
    "worst texture": "Największa tekstura",
    "worst perimeter": "Największy obwód",
    "worst area": "Największa powierzchnia",
    "worst smoothness": "Największa gładkość",
    "worst compactness": "Największa zwartość",
    "worst concavity": "Największa wklęsłość",
    "worst concave points": "Największa liczba punktów wklęsłych",
    "worst symmetry": "Największa symetria",
    "worst fractal dimension": "Największy wymiar fraktalny",
    "target": "Diagnoza"
}


data = load_breast_cancer()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = 1 - data.target

df["diagnosis"] = df["target"].map({
    0: "Łagodny",
    1: "Złośliwy"
})


print("Pierwsze 5 wierszy:")
print(df.head())

print("\nWymiary danych:")
print(df.shape)

print("\nInformacje o danych:")
df.info()

print("\nLiczba brakujących wartości:")
print(df.isnull().sum().sum())

print("\nLiczba przypadków dla każdej diagnozy:")
print(df["diagnosis"].value_counts())

print("\nProcentowy udział klas:")
print(df["diagnosis"].value_counts(normalize=True) * 100)


diagnosis_counts = (
    df["diagnosis"]
    .value_counts()
    .reindex(["Złośliwy", "Łagodny"])
)

plt.figure(figsize=(7, 5))

bars = plt.bar(
    diagnosis_counts.index,
    diagnosis_counts.values,
    color=["#C62828", "#F48FB1"]
)

plt.title("Rozkład diagnoz w zbiorze danych", fontsize=14)
plt.xlabel("Diagnoza")
plt.ylabel("Liczba przypadków")

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 5,
        str(int(height)),
        ha="center",
        fontsize=11
    )

plt.tight_layout()
plt.savefig(
    images_dir / "diagnosis_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


features = [
    "mean radius",
    "mean texture",
    "mean area",
    "mean concavity"
]

print("\nŚrednie wartości wybranych cech:")
print(df.groupby("diagnosis")[features].mean())


fig, axes = plt.subplots(2, 2, figsize=(12, 8))

titles = [
    "Średni promień guza",
    "Średnia tekstura",
    "Średnia powierzchnia guza",
    "Średnia wklęsłość"
]

y_labels = [
    "Średni promień",
    "Średnia tekstura",
    "Średnia powierzchnia",
    "Średnia wklęsłość"
]

for ax, feature, title, y_label in zip(
    axes.flatten(),
    features,
    titles,
    y_labels
):
    malignant = df[df["diagnosis"] == "Złośliwy"][feature]
    benign = df[df["diagnosis"] == "Łagodny"][feature]

    box = ax.boxplot(
        [malignant, benign],
        tick_labels=["Złośliwy", "Łagodny"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5
        )
    )

    colors = ["#C62828", "#F48FB1"]

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(
    images_dir / "feature_boxplots.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


numeric_df = df.drop(columns=["diagnosis"])

correlation_matrix = numeric_df.corr()

target_correlation = (
    correlation_matrix["target"]
    .drop("target")
    .sort_values(ascending=False)
)

print("\nKorelacja cech z diagnozą:")
print(target_correlation)

top_features = (
    target_correlation
    .abs()
    .sort_values(ascending=False)
    .head(10)
    .index
)

print("\n10 cech najmocniej skorelowanych z diagnozą:")
print(top_features)


selected_columns = list(top_features) + ["target"]

selected_corr = df[selected_columns].corr()

labels_pl = [
    feature_names_pl[column]
    for column in selected_columns
]

plt.figure(figsize=(12, 10))

plt.imshow(
    selected_corr,
    cmap="RdPu",
    aspect="auto",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Współczynnik korelacji")

plt.xticks(
    range(len(labels_pl)),
    labels_pl,
    rotation=90
)

plt.yticks(
    range(len(labels_pl)),
    labels_pl
)

plt.title(
    "Macierz korelacji najważniejszych cech",
    fontsize=14
)

plt.tight_layout()
plt.savefig(
    images_dir / "correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


X = df.drop(columns=["target", "diagnosis"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nRozmiar zbioru treningowego:")
print(X_train.shape)

print("\nRozmiar zbioru testowego:")
print(X_test.shape)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


log_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

log_model.fit(
    X_train_scaled,
    y_train
)

y_pred = log_model.predict(X_test_scaled)

y_prob = log_model.predict_proba(
    X_test_scaled
)[:, 1]


accuracy_log = accuracy_score(
    y_test,
    y_pred
)

roc_auc_log = roc_auc_score(
    y_test,
    y_prob
)

recall_log = recall_score(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n=========================")
print("REGRESJA LOGISTYCZNA")
print("=========================")

print("\nAccuracy:")
print(round(accuracy_log, 4))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Łagodny", "Złośliwy"]
    )
)

print("\nMacierz pomyłek:")
print(cm)

print("\nROC-AUC:")
print(round(roc_auc_log, 4))

print("\nRecall dla zmian złośliwych:")
print(round(recall_log, 4))


fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(
    cm,
    cmap="RdPu"
)

plt.colorbar(im)

classes = ["Łagodny", "Złośliwy"]

ax.set_xticks(range(2))
ax.set_yticks(range(2))

ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

ax.set_xlabel("Przewidywana diagnoza")
ax.set_ylabel("Rzeczywista diagnoza")

ax.set_title(
    "Macierz pomyłek – regresja logistyczna"
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold"
        )

plt.tight_layout()
plt.savefig(
    images_dir / "confusion_matrix_logistic.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    color="#C62828",
    label=f"Regresja logistyczna (AUC = {roc_auc_log:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Klasyfikacja losowa"
)

plt.xlabel("Odsetek fałszywie dodatnich")
plt.ylabel("Czułość")
plt.title("Krzywa ROC – regresja logistyczna")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    images_dir / "roc_logistic.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


coefficients = pd.DataFrame({
    "Cecha": X.columns,
    "Współczynnik": log_model.coef_[0]
})

coefficients["Znaczenie"] = (
    coefficients["Współczynnik"].abs()
)

coefficients = coefficients.sort_values(
    by="Znaczenie",
    ascending=False
)

print("\nNajważniejsze cechy w regresji logistycznej:")
print(coefficients.head(10))


top_coefficients = coefficients.head(10).copy()

top_coefficients["Cecha PL"] = (
    top_coefficients["Cecha"]
    .map(feature_names_pl)
)

top_coefficients = top_coefficients.sort_values(
    by="Współczynnik",
    ascending=True
)

colors = [
    "#C62828" if value > 0 else "#F48FB1"
    for value in top_coefficients["Współczynnik"]
]

plt.figure(figsize=(9, 6))

plt.barh(
    top_coefficients["Cecha PL"],
    top_coefficients["Współczynnik"],
    color=colors
)

plt.axvline(
    x=0,
    color="black",
    linewidth=1
)

plt.xlabel("Współczynnik regresji logistycznej")
plt.ylabel("Cecha")
plt.title("Wpływ cech na przewidywanie zmiany złośliwej")

plt.text(
    0.02,
    0.02,
    "Czerwony → w kierunku zmiany złośliwej\nRóżowy → w kierunku zmiany łagodnej",
    transform=plt.gca().transAxes,
    fontsize=9
)

plt.tight_layout()

plt.savefig(
    images_dir / "logistic_feature_coefficients.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

y_pred_rf = rf_model.predict(
    X_test
)

y_prob_rf = rf_model.predict_proba(
    X_test
)[:, 1]

accuracy_rf = accuracy_score(
    y_test,
    y_pred_rf
)

roc_auc_rf = roc_auc_score(
    y_test,
    y_prob_rf
)

recall_rf = recall_score(
    y_test,
    y_pred_rf
)

cm_rf = confusion_matrix(
    y_test,
    y_pred_rf
)


print("\n=========================")
print("RANDOM FOREST")
print("=========================")

print("\nAccuracy:")
print(round(accuracy_rf, 4))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred_rf,
        target_names=["Łagodny", "Złośliwy"]
    )
)

print("\nMacierz pomyłek:")
print(cm_rf)

print("\nROC-AUC:")
print(round(roc_auc_rf, 4))

print("\nRecall dla zmian złośliwych:")
print(round(recall_rf, 4))


fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(
    cm_rf,
    cmap="RdPu"
)

plt.colorbar(im)

classes = ["Łagodny", "Złośliwy"]

ax.set_xticks(range(2))
ax.set_yticks(range(2))

ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

ax.set_xlabel("Przewidywana diagnoza")
ax.set_ylabel("Rzeczywista diagnoza")

ax.set_title(
    "Macierz pomyłek – Random Forest"
)

for i in range(cm_rf.shape[0]):
    for j in range(cm_rf.shape[1]):
        ax.text(
            j,
            i,
            cm_rf[i, j],
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold"
        )

plt.tight_layout()
plt.savefig(
    images_dir / "confusion_matrix_random_forest.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


feature_importance_rf = pd.DataFrame({
    "Cecha": X.columns,
    "Ważność": rf_model.feature_importances_
})

feature_importance_rf = (
    feature_importance_rf
    .sort_values(
        by="Ważność",
        ascending=False
    )
)

print("\n10 najważniejszych cech według Random Forest:")
print(feature_importance_rf.head(10))


top_rf_features = (
    feature_importance_rf
    .head(10)
    .copy()
)

top_rf_features["Cecha PL"] = (
    top_rf_features["Cecha"]
    .map(feature_names_pl)
)

top_rf_features = top_rf_features.sort_values(
    by="Ważność",
    ascending=True
)

plt.figure(figsize=(9, 6))

plt.barh(
    top_rf_features["Cecha PL"],
    top_rf_features["Ważność"],
    color="#C62828"
)

plt.xlabel("Ważność cechy")
plt.ylabel("Cecha")
plt.title("Najważniejsze cechy według Random Forest")

plt.tight_layout()
plt.savefig(
    images_dir / "random_forest_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


comparison = pd.DataFrame({
    "Model": [
        "Regresja logistyczna",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_log,
        accuracy_rf
    ],
    "ROC-AUC": [
        roc_auc_log,
        roc_auc_rf
    ],
    "Recall – złośliwy": [
        recall_log,
        recall_rf
    ]
})

print("\n=========================")
print("PORÓWNANIE MODELI")
print("=========================")

print(comparison.round(4))


x = range(len(comparison))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    [i - width / 2 for i in x],
    comparison["Accuracy"],
    width=width,
    label="Accuracy",
    color="#F48FB1"
)

plt.bar(
    [i + width / 2 for i in x],
    comparison["ROC-AUC"],
    width=width,
    label="ROC-AUC",
    color="#C62828"
)

plt.xticks(
    list(x),
    comparison["Model"]
)

plt.ylim(0.85, 1.01)

plt.ylabel("Wartość metryki")
plt.title("Porównanie modeli klasyfikacyjnych")

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    images_dir / "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


print("\nRecall dla zmian złośliwych:")
print(
    "Regresja logistyczna:",
    round(recall_log, 4)
)
print(
    "Random Forest:",
    round(recall_rf, 4)
)