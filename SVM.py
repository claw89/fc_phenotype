import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score


exps = [
    ['AM', 'EP'], 
    ['EPI', 'EP'],
    ['AM', 'EPI'], 
    ['AM', 'EBM'], 
    ['AM', 'FC/SC'], 
    ['EBM', 'FC/SC'],
    ['FC', 'SC']
    ]

names = {
    'AM': 'Adult Man',
    'EP': 'Elvis Presley',
    'EPI': 'Elvis Presley Impersonator',
    'EBM': 'Elderly Bearded Man',
    'FC': 'Father Christmas',
    'SC': 'Santa Claus',
    'S': 'Santa'
}

subfig_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']


def main():

    fig, ax = plt.subplots(3, 3, figsize=(15, 15))

    ax = ax.ravel()

    for i, (control, test) in enumerate(exps):
      df = pd.read_csv('data/data.csv')
      if len(test.split('/')) < 2:
        exp_name = f'{names[test]} vs\n{names[control]}'
      else:
        test_parts = test.split('/')
        exp_name = f'{"/".join([names[test_part] for test_part in test_parts])} vs\n{names[control]}'

        for test_part in test_parts:
          df.group.replace(test_parts, test, inplace=True)

      exp_name = exp_name.replace('Father Christmas/Santa Claus', 'Total Father Christmas')
      print(f'Experiment {i}: {exp_name}')

      df = df[df.group.isin([control, test])]
      df_train = df[df.split == 'train']
      df_test = df[df.split == 'test']

      X_train = df_train.drop(['group', 'split'], axis=1).values
      X_test = df_test.drop(['group', 'split'], axis=1).values
      y_train = df_train.group.map({control: 0, test: 1}).values
      y_test = df_test.group.map({control: 0, test: 1}).values

      print(f'Training samples: {X_train.shape[0]}')
      print(f'Testing samples: {X_test.shape[0]}')

      parameters = {
          'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 
          'C':[0.1, 1, 10, 100],
          'degree': [2, 3, 4, 5],
          'gamma': ('scale', 'auto'),
          'coef0': [0.0, 1.0, 10.0],
          'shrinking': (True, False),
          'tol': [0.0001, 0.001, 0.01, 0.1]
          }
      svc = SVC(probability=True)
      clf = GridSearchCV(svc, parameters, verbose=1)
      clf.fit(X_train, y_train)

      if len(test.split('/')) < 2:
        with open(f'SVM/{control}_{test}', 'wb') as file:
          pickle.dump(clf, file)
      else:
        with open(f'SVM/{control}_{"".join(test_parts)}', 'wb') as file:
          pickle.dump(clf, file)

      y_score = clf.decision_function(X_test)
      fpr, tpr, _ = roc_curve(y_test, y_score)
      print(f'Accuracy: {clf.best_score_:.4f}')
      print(f'Precision: {precision_score(y_test, clf.predict(X_test)):.4f}')
      print(f'Recall: {recall_score(y_test, clf.predict(X_test)):.4f}')
      print(f'AUC: {auc(fpr, tpr):.4f}')
      ax[i].plot(fpr, tpr, color="darkorange", label=f'ROC curve (AUC = {auc(fpr, tpr):.4f})')
      ax[i].set_title(f'{subfig_labels[i]}) {exp_name}')
      ax[i].plot([0, 1], [0, 1], color="black", linestyle="--")
      ax[i].legend(loc='lower right')
      print('\n')

    ax[7].axis('off')
    ax[8].axis('off')
    plt.show()


if __name__ == "__main__":
    main()
