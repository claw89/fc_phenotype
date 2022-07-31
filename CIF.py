from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os

def compute_distance(df, file1, file2):
    return np.linalg.norm(df.loc[file1].values - df.loc[file2].values) 

def compute_closest_rank(distances, pos_indexes, pos_index):
    rank_df = pd.DataFrame({pos_index: distances[pos_index]}).sort_values(pos_index).reset_index()
    return min([v for v in rank_df[rank_df['index'].isin(pos_indexes)].index.values if v != 0])

def compare_facial_phenotypes(controls, targets, df, n_permutations=10_000):
  df = df[df.group.isin([controls, targets])].reset_index()
  df_dist = df.drop('group', axis=1)

  distances = np.zeros((df_dist.index.shape[0], df_dist.index.shape[0]))

  print('Computing pairwise distances')
  for i, file1 in enumerate(tqdm(df_dist.index)):
      for j, file2 in enumerate(df_dist.index):
          if i <= j:
              distances[i][j] = compute_distance(df_dist, file1, file2)

  # Copy distance calculations accross the matrix diagonal
  for i in range(distances.shape[0]):
      for j in range(distances.shape[1]):
          distances[j][i] = distances[i][j]

  N_neg = df[df.group == controls].shape[0]
  N_pos = df[df.group == targets].shape[0]

  exp_rank = 1 + sum([(1 - (j+1)/(N_neg + 1))**(N_pos - 1) for j in range(N_neg)])

  pos_indexes = df[df.group == targets].index.values

  obs_rank = np.mean([compute_closest_rank(distances, pos_indexes, pos_index) for pos_index in pos_indexes])

  CIF_target = exp_rank/obs_rank

  CIF_rand = []
  print('Running permutation test')
  for _ in tqdm(range(n_permutations)):

      df_permuted = df.copy()
      df_permuted['group'] = np.random.permutation(df_permuted.group.values)
      
      pos_indexes = df_permuted[df_permuted.group == targets].index.values

      obs_rank = np.mean([compute_closest_rank(distances, pos_indexes, pos_index) for pos_index in pos_indexes])
      
      CIF_rand.append(exp_rank/obs_rank)
    
  p = len([cif for cif in CIF_rand if cif > CIF_target]) / len(CIF_rand)

  return CIF_target, CIF_rand, p

def main():
    df = pd.read_csv('data/data.csv')
    df = df.drop('split', axis=1)

    CIF_target, CIF_rand, p = compare_facial_phenotypes('AM', 'EP', df)
    print(f'Elvis Presley vs Adult Man (control): CIF = {CIF_target}, p = {p}')

    CIF_target, CIF_rand, p = compare_facial_phenotypes('EPI', 'EP', df)
    print(f'Elvis Presley vs Elvis Presley Impersonator (control): CIF = {CIF_target}, p = {p}')

    CIF_target, CIF_rand, p = compare_facial_phenotypes('AM', 'EPI', df)
    print(f'Elvis Presley Impersonator vs Adult Man (control): CIF = {CIF_target}, p = {p}')

    CIF_target, CIF_rand, p = compare_facial_phenotypes('AM', 'FC/SC', df.replace('FC', 'FC/SC').replace('SC', 'FC/SC'))
    print(f'Total Father Christmas vs Adult Man (control): CIF = {CIF_target}, p = {p}')

    CIF_target, CIF_rand, p = compare_facial_phenotypes('AM', 'EBM', df)
    print(f'Elderly Bearded Man vs Adult Man (control): CIF = {CIF_target}, p = {p}')

    CIF_target, CIF_rand, p = compare_facial_phenotypes('EBM', 'FC/SC', df.replace('FC', 'FC/SC').replace('SC', 'FC/SC'))
    print(f'Total Father Christmas vs Elderly Bearded Man (control): CIF = {CIF_target}, p = {p}')
