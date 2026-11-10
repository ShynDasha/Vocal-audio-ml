"""Train on cached features using a reproducible, singer-disjoint holdout."""
import hashlib
import json
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import FEATURES, OUTPUT_DIR, SEED


def split_dataset(dataset):
    if dataset.technique.nunique() < 2 or dataset.singer.nunique() < 2:
        raise ValueError('Потрібні щонайменше дві техніки та два виконавці.')
    classes = set(dataset.technique)
    # Choose by label coverage only, never by model performance.
    splitter = GroupShuffleSplit(n_splits=100, test_size=0.25, random_state=SEED)
    for train, test in splitter.split(dataset, groups=dataset.singer):
        if set(dataset.iloc[train].technique) == classes == set(dataset.iloc[test].technique):
            return train, test
    raise ValueError('Не вдалося представити всі техніки в обох частинах без перетину виконавців. Збільште вибірку; перевірте покриття технік за виконавцями.')


def main():
    if not FEATURES.exists():
        raise FileNotFoundError('Спочатку запустіть audio_features.py')
    metadata = json.loads(FEATURES.with_suffix('.json').read_text())
    if hashlib.sha256(FEATURES.read_bytes()).hexdigest() != metadata['features_sha256']:
        raise ValueError('Таблиця ознак не відповідає метаданим. Повторіть обчислення ознак.')
    dataset = pd.read_csv(FEATURES)
    columns = [c for c in dataset if c not in {'path', 'technique', 'singer', 'sample_type'}]
    if not np.isfinite(dataset[columns].to_numpy()).all():
        raise ValueError('Ознаки містять NaN або нескінченні значення.')
    train, test = split_dataset(dataset)
    X, y = dataset[columns], dataset.technique
    report_dir, model_dir = OUTPUT_DIR / 'reports', OUTPUT_DIR / 'models'
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    split = dataset[['path', 'singer', 'technique']].copy()
    split['split'] = 'train'
    split.loc[test, 'split'] = 'test'
    split.to_csv(report_dir / 'split.csv', index=False)
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight='balanced'),
        'SVM': make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight='balanced')),
    }
    results, reports = [], {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X.iloc[train], y.iloc[train])
        train_time = time.perf_counter() - start
        start = time.perf_counter()
        predictions = model.predict(X.iloc[test])
        predict_time = time.perf_counter() - start
        results.append({'Model': name, 'Accuracy': accuracy_score(y.iloc[test], predictions),
                        'Macro F1': f1_score(y.iloc[test], predictions, average='macro', zero_division=0),
                        'Train time (s)': train_time,
                        'Predict time (ms/sample)': 1000 * predict_time / len(test)})
        reports[name] = classification_report(y.iloc[test], predictions, output_dict=True, zero_division=0)
        print(name, classification_report(y.iloc[test], predictions, zero_division=0), sep='\n')
    results = pd.DataFrame(results)
    results.to_csv(report_dir / 'metrics.csv', index=False)
    (report_dir / 'classification_reports.json').write_text(json.dumps(reports, indent=2))
    best_name = results.loc[results['Macro F1'].idxmax(), 'Model']
    results.plot(x='Model', y=['Accuracy', 'Macro F1'], kind='bar', rot=0, ylim=(0, 1))
    plt.tight_layout()
    plt.savefig(report_dir / 'model_comparison.png')
    plt.close()
    ConfusionMatrixDisplay.from_estimator(models[best_name], X.iloc[test], y.iloc[test], xticks_rotation=90)
    plt.tight_layout()
    plt.savefig(report_dir / 'confusion_matrix.png')
    plt.close()
    joblib.dump({'model': models[best_name], 'features': columns,
                 'sample_rate': metadata['sample_rate'], 'duration': metadata['duration'],
                 'feature_metadata': metadata, 'seed': SEED, 'model_name': best_name,
                 'train_singers': sorted(dataset.iloc[train].singer.unique()),
                 'test_singers': sorted(dataset.iloc[test].singer.unique())}, model_dir / 'vocal_model.joblib')
    print(results.to_string(index=False))
    print('Найкраща модель цього експерименту:', best_name)
    print('Вибір моделі за цим holdout є розвідувальним; фінальна оцінка потребує окремого тесту.')


if __name__ == '__main__':
    main()
