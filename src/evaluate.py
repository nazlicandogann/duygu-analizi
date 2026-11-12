import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

ETIKETLER = ['Pozitif', 'Negatif', 'Nötr']

def evaluate(dosya, gorsel_klasoru='results/figures'):
    df = pd.read_csv(dosya, encoding='utf-8')
    y_true = df['true']
    y_pred = df['pred']

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=ETIKETLER, average='macro', zero_division=0)
    
def evaluate(dosya, gorsel_klasoru='results/figures'):
    df = pd.read_csv(dosya, encoding='utf-8')
    y_true = df['true']
    y_pred = df['pred']

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=ETIKETLER, average='macro', zero_division=0)

    hatalar = {}
    for etiket in ETIKETLER:
        hatalar[etiket] = int(((y_true == etiket) & (y_pred != etiket)).sum())

    gecersiz = int((~y_pred.isin(ETIKETLER)).sum())

    cm = confusion_matrix(y_true, y_pred, labels=ETIKETLER)

    os.makedirs(gorsel_klasoru, exist_ok=True)
    ad = os.path.splitext(os.path.basename(dosya))[0].replace('preds_', 'cm_')
    gorsel_yolu = os.path.join(gorsel_klasoru, ad + '.png')

    ConfusionMatrixDisplay(cm, display_labels=ETIKETLER).plot(cmap='Blues', values_format='d')
    plt.title(ad)
    plt.tight_layout()
    plt.savefig(gorsel_yolu, dpi=150)
    plt.close()

    return {
        'dosya': os.path.basename(dosya),
        'n': len(df),
        'accuracy': round(accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'hatalar': hatalar,
        'gecersiz': gecersiz,
        'gorsel': gorsel_yolu,
    } 


if __name__ == '__main__':
    sonuc = evaluate(sys.argv[1])
    for anahtar, deger in sonuc.items():
        print(f'{anahtar}: {deger}')
