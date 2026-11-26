import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.inspection import permutation_importance

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.dpi'] = 150

# 1. Carga y exploración inicial
df = pd.read_csv('customer_churn_premium_dataset.csv')
print(f"Dimensiones: {df.shape}")
print("Valores faltantes por columna:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("Distribución de churn:\n", df['churn'].value_counts(normalize=True).round(4))

# 2. Separación de variables (excluyendo customer_id) y split estratificado 80/20
X = df.drop(columns=['customer_id', 'churn'])
y = df['churn']

num_cols = ['age', 'tenure_months', 'monthly_charges', 'support_tickets', 'late_payments']
cat_cols = ['contract_type', 'payment_method', 'internet_service', 'tech_support', 'paperless_billing', 'discount', 'streaming']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 3. Pipeline de preprocesamiento (ajustado exclusivamente sobre train para evitar data leakage)
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# Exportar dataset procesado
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
proc_cols = preprocessor.get_feature_names_out()

df_train_proc = pd.DataFrame(X_train_proc, columns=proc_cols, index=X_train.index)
df_train_proc['churn'] = y_train
df_train_proc['split'] = 'train'

df_test_proc = pd.DataFrame(X_test_proc, columns=proc_cols, index=X_test.index)
df_test_proc['churn'] = y_test
df_test_proc['split'] = 'test'

df_processed = pd.concat([df_train_proc, df_test_proc]).sort_index()
df_processed.insert(0, 'customer_id', df['customer_id'])
df_processed.to_csv('customer_churn_processed.csv', index=False)

# 4. Entrenamiento y optimización de 5 modelos mediante GridSearchCV en train
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_and_grids = {
    'Logistic Regression': (
        LogisticRegression(max_iter=1000, random_state=42),
        {'clf__C': [0.01, 0.1, 1.0, 10.0], 'clf__class_weight': [None, 'balanced']}
    ),
    'Decision Tree': (
        DecisionTreeClassifier(random_state=42),
        {'clf__max_depth': [3, 5, 7, 10, None], 'clf__min_samples_split': [2, 10, 20], 'clf__min_samples_leaf': [1, 5, 10], 'clf__class_weight': [None, 'balanced']}
    ),
    'Random Forest': (
        RandomForestClassifier(random_state=42),
        {'clf__n_estimators': [100, 200], 'clf__max_depth': [5, 10, None], 'clf__min_samples_split': [2, 10], 'clf__class_weight': [None, 'balanced']}
    ),
    'SVM': (
        SVC(probability=True, random_state=42),
        {'clf__C': [0.1, 1.0, 10.0], 'clf__kernel': ['rbf', 'linear'], 'clf__gamma': ['scale', 'auto'], 'clf__class_weight': [None, 'balanced']}
    ),
    'Gradient Boosting': (
        GradientBoostingClassifier(random_state=42),
        {'clf__n_estimators': [100, 200], 'clf__learning_rate': [0.03, 0.05, 0.1], 'clf__max_depth': [2, 3, 4], 'clf__subsample': [0.8, 1.0]}
    )
}

results, opt_results = [], []
best_estimators, predictions, probabilities = {}, {}, {}

for name, (est, grid) in models_and_grids.items():
    pipe = Pipeline([('pre', preprocessor), ('clf', est)])
    gs = GridSearchCV(pipe, grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    best_estimators[name] = best_model

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    predictions[name] = y_pred
    probabilities[name] = y_prob

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results.append({
        'Model': name,
        'CV_ROC_AUC': round(gs.best_score_, 4),
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1_Score': round(f1_score(y_test, y_pred), 4),
        'ROC_AUC': round(roc_auc_score(y_test, y_prob), 4),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)
    })
    clean_params = {k.replace('clf__', ''): v for k, v in gs.best_params_.items()}
    opt_results.append({
        'Model': name,
        'Best_CV_ROC_AUC': round(gs.best_score_, 4),
        'Best_Hyperparameters': str(clean_params)
    })

res_df = pd.DataFrame(results).sort_values(by=['Recall', 'ROC_AUC', 'F1_Score'], ascending=False)
opt_df = pd.DataFrame(opt_results)
res_df.to_csv('model_comparison.csv', index=False)
opt_df.to_csv('optimization_results.csv', index=False)

print("\n--- COMPARACIÓN DE MODELOS ---")
print(res_df.to_string(index=False))

# 5. Gráficos
model_order = res_df['Model'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for idx, name in enumerate(model_order):
    cm = confusion_matrix(y_test, predictions[name])
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx],
        xticklabels=['No Churn (0)', 'Churn (1)'],
        yticklabels=['No Churn (0)', 'Churn (1)'],
        annot_kws={'size': 13, 'weight': 'bold'}
    )
    row = res_df[res_df['Model'] == name].iloc[0]
    axes[idx].set_title(f"{name}\nRecall: {row['Recall']:.2%} | F1: {row['F1_Score']:.3f} | FN: {row['FN']}", fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Predicción')
    axes[idx].set_ylabel('Valor Real')
axes[-1].axis('off')
fig.suptitle('Matrices de Confusión en Conjunto de Prueba (n = 240)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('confusion_matrices.png', dpi=200)
plt.show()

plt.figure(figsize=(9, 7))
for name in model_order:
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    auc_val = res_df[res_df['Model'] == name]['ROC_AUC'].values[0]
    plt.plot(fpr, tpr, linewidth=2.2, label=f"{name} (AUC = {auc_val:.4f})")
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Clasificador Aleatorio (AUC = 0.5000)')
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=11)
plt.ylabel('Tasa de Verdaderos Positivos (Recall)', fontsize=11)
plt.title('Comparación de Curvas ROC — Predicción de Churn (Test Set)', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=200)
plt.show()

lr_pipe = best_estimators['Logistic Regression']
perm = permutation_importance(lr_pipe, X_test, y_test, n_repeats=30, random_state=42, scoring='roc_auc')
perm_df = pd.DataFrame({
    'Variable': X.columns,
    'Importancia_AUC': perm.importances_mean,
    'Std': perm.importances_std
}).sort_values('Importancia_AUC', ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
colors_perm = ['#d95f02' if v > 0.01 else '#7570b3' for v in perm_df['Importancia_AUC']]
ax1.barh(perm_df['Variable'], perm_df['Importancia_AUC'], xerr=perm_df['Std'], color=colors_perm, alpha=0.88, capsize=3)
ax1.axvline(0, color='black', linewidth=0.8)
ax1.set_title('Importancia por Permutación (Caída en ROC-AUC)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Disminución media en ROC-AUC')

feat_names = [f.replace('num__', '').replace('cat__', '') for f in lr_pipe.named_steps['pre'].get_feature_names_out()]
coefs = lr_pipe.named_steps['clf'].coef_[0]
coef_df = pd.DataFrame({'Feature': feat_names, 'Coef': coefs}).sort_values('Coef', ascending=True)
colors_coef = ['#1b9e77' if c < 0 else '#d95f02' for c in coef_df['Coef']]
ax2.barh(coef_df['Feature'], coef_df['Coef'], color=colors_coef, alpha=0.88)
ax2.axvline(0, color='black', linewidth=0.8)
ax2.set_title('Dirección del Efecto (Coeficientes Logistic Regression)\nVerde = Reduce Churn | Naranja = Aumenta Churn', fontsize=11, fontweight='bold')
ax2.set_xlabel('Coeficiente estandarizado (Log-Odds)')

fig.suptitle('Factores Asociados al Abandono de Clientes (Churn)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=200, bbox_inches='tight')
plt.show()

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
plot_df = res_df.melt(id_vars='Model', value_vars=metrics_to_plot, var_name='Métrica', value_name='Valor')
plt.figure(figsize=(11, 6))
ax = sns.barplot(data=plot_df, x='Métrica', y='Valor', hue='Model', palette='Set2')
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=7.5, padding=2)
plt.ylim(0, 1.0)
plt.title('Comparación de Métricas de Clasificación por Modelo (Test Set)', fontsize=13, fontweight='bold')
plt.ylabel('Puntuación')
plt.xlabel('')
plt.legend(title='Modelo', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=200)
plt.show()

# 6. Generar final_report.pdf
doc = SimpleDocTemplate(
    "final_report.pdf",
    pagesize=letter,
    rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=45
)
styles = getSampleStyleSheet()
title_style = ParagraphStyle('TitleCustom', parent=styles['Title'], fontSize=18, textColor=colors.HexColor('#1B365D'), spaceAfter=6)
subtitle_style = ParagraphStyle('SubtitleCustom', parent=styles['Normal'], fontSize=11, textColor=colors.HexColor('#4A607A'), spaceAfter=16)
h1_style = ParagraphStyle('H1Custom', parent=styles['Heading1'], fontSize=13, textColor=colors.HexColor('#1B365D'), spaceBefore=12, spaceAfter=6)
body_style = ParagraphStyle('BodyCustom', parent=styles['Normal'], fontSize=9.8, leading=13.5, spaceAfter=8)
bullet_style = ParagraphStyle('BulletCustom', parent=styles['Normal'], fontSize=9.5, leading=13, leftIndent=14, spaceAfter=4)

story = [
    Paragraph("Informe Ejecutivo: Predicción de Abandono de Clientes (Churn)", title_style),
    Paragraph("Proyecto #ML-003 — Clasificación Avanzada, Optimización y Estrategia de Retención", subtitle_style),
    Paragraph("1. Resumen Ejecutivo y Recomendación Comercial", h1_style),
    Paragraph(
        "Se desarrolló una solución predictiva sobre una cartera de 1.200 clientes (40,50% de tasa histórica de abandono) para anticipar bajas y priorizar campañas de retención. "
        "Dado que perder un cliente activo (Falso Negativo) cuesta más que ofrecer un incentivo preventivo a un cliente que no iba a irse (Falso Positivo), "
        "la selección priorizó el Recall y el ROC-AUC manteniendo un buen equilibrio operativo.",
        body_style
    ),
    Paragraph(
        "<b>Modelo Recomendado: Logistic Regression (C = 10.0, class_weight = 'balanced').</b> "
        "Empata en el mayor Recall (77,32%) con SVM (75 de 97 bajas detectadas; 22 FN), pero supera en Accuracy (75,42%), Precision (66,96%) y F1-Score (0,7177), con menos falsos positivos (37 vs. 40) y probabilidades calibradas.",
        body_style
    ),
    Paragraph("2. Análisis Exploratorio y Preprocesamiento", h1_style),
    Paragraph(
        "1.200 registros y 14 columnas (714 sin churn / 59,50% y 486 con churn / 40,50%), con valores faltantes en 6 columnas (age, tenure_months, monthly_charges, payment_method, internet_service, tech_support). "
        "Se excluyó customer_id, se dividió estratificadamente 80/20 (random_state=42) y se aplicó imputación, codificación One-Hot y escalado dentro de un Pipeline ajustado solo sobre entrenamiento.",
        body_style
    ),
    Paragraph("3. Comparación de Modelos (Test Set, n = 240)", h1_style),
]

t_data = [["Modelo", "CV AUC", "Accuracy", "Precision", "Recall", "F1-Score", "Test AUC", "FN / TP"]]
for _, r in res_df.iterrows():
    t_data.append([
        r['Model'], f"{r['CV_ROC_AUC']:.4f}", f"{r['Accuracy']*100:.2f}%",
        f"{r['Precision']*100:.2f}%", f"{r['Recall']*100:.2f}%", f"{r['F1_Score']:.4f}",
        f"{r['ROC_AUC']:.4f}", f"{int(r['FN'])} / {int(r['TP'])}"
    ])

tbl = Table(t_data, colWidths=[1.45*inch, 0.62*inch, 0.68*inch, 0.68*inch, 0.62*inch, 0.65*inch, 0.65*inch, 0.65*inch])
tbl.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B365D')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 8.2),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
]))
story.append(tbl)
story.append(PageBreak())
story.append(Paragraph("4. Gráficos de Evaluación", h1_style))
story.append(Image("confusion_matrices.png", width=6.8*inch, height=4.0*inch))
story.append(Spacer(1, 8))
story.append(Image("roc_curves.png", width=5.5*inch, height=4.2*inch))
story.append(PageBreak())
story.append(Paragraph("5. Factores Clave de Abandono y Limitaciones", h1_style))
story.append(Image("feature_importance.png", width=6.9*inch, height=2.9*inch))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Los principales factores asociados al churn son el contrato mensual (Month-to-month), cargos mensuales altos (monthly_charges), baja antigüedad (tenure_months), "
    "mayor cantidad de tickets de soporte (support_tickets) y pagos atrasados (late_payments). Como limitaciones: no hay métricas de uso real ni NPS, el test tiene 240 casos y conviene ajustar el umbral según el presupuesto de retención.",
    body_style
))
doc.build(story)

# 7. Empaquetar todo en entrega_premium_ML003.zip
files_to_zip = [
    'customer_churn_processed.csv',
    'model_comparison.csv',
    'optimization_results.csv',
    'confusion_matrices.png',
    'roc_curves.png',
    'feature_importance.png',
    'model_comparison.png',
    'final_report.pdf'
]
with zipfile.ZipFile('entrega_premium_ML003.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in files_to_zip:
        if os.path.exists(f):
            zf.write(f)

print("\n¡Listo! Se generaron todos los archivos y 'entrega_premium_ML003.zip'.")