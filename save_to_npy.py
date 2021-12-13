from ast import literal_eval
import pandas as pd
import numpy as np

def save_X(train, val, test, model, model_suffix, use_as_is_col, list_cols):
    train_data = train[use_as_is_col].to_numpy()
    val_data = val[use_as_is_col].to_numpy()
    test_data = test[use_as_is_col].to_numpy()
    for c in list_cols:
        train_data = np.append(train_data, np.array(train[c].tolist(), dtype=np.float32), axis=1)
        val_data = np.append(val_data, np.array(val[c].tolist(), dtype=np.float32), axis=1)
        test_data = np.append(test_data, np.array(test[c].tolist(), dtype=np.float32), axis=1)

    np.save(f'./numpy_data/{model}{model_suffix}_train_X', train_data)
    np.save(f'./numpy_data/{model}{model_suffix}_val_X', val_data)
    np.save(f'./numpy_data/{model}{model_suffix}_test_X', test_data)

use_as_is_col = ['Add', 'Modify', 'Delete', 'Organization', 'Word-Usage/Clarity', 'Claims/Ideas',
    'Warrant/Reasoning/Backing', 'Rebuttal/Reservation', 'Evidence', 'Precision', 'General Content Development']
rubrics = ['d_Prompt', 'd_Thesis', 'd_Claims', 'd_Evidence', 'd_Reasoning', 'd_Organization', 
    'd_Rebuttal', 'd_Precision', 'd_Fluency', 'd_Coventions']


list_cols_sbert = ['prev_para_embed_sbert', 'prev_sent_embed_sbert', 'target_embed_sbert', 'fol_sent_embed_sbert', 'fol_para_embed_sbert']
converters = {c: literal_eval for c in list_cols_sbert}
df = pd.read_csv('edits_sbert_embedding.csv', converters=converters)

#for c in list_cols_sbert:
#    df[c] = df[c].str.strip('[]').str.strip(' ').str.replace('\n', '').str.replace('  ', ' ').str.replace(',', '').str.split(' ')
df = df[df['Conventions/Grammar/Spelling'] == 0]

train = df[(df['val_data'] == 0) & (df['test_data'] == 0)].copy()
val = df[df['val_data'] == 1].copy()
test = df[df['test_data'] == 1].copy()
save_X(train, val, test, 'sbert', '', use_as_is_col, list_cols_sbert)

for rubric in rubrics:
    np.save(f'./numpy_data/{rubric}_train_y', train[rubric].to_numpy())
    np.save(f'./numpy_data/{rubric}_val_y', val[rubric].to_numpy())
    np.save(f'./numpy_data/{rubric}_test_y', test[rubric].to_numpy())

    np.save(f'./numpy_data/{rubric}_bin_train_y', (train[rubric] > 0).astype(int).to_numpy())
    np.save(f'./numpy_data/{rubric}_bin_val_y', (val[rubric] > 0).astype(int).to_numpy())
    np.save(f'./numpy_data/{rubric}_bin_test_y', (test[rubric] > 0).astype(int).to_numpy())
