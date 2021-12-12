from ast import literal_eval
import numpy as np
import pandas as pd
from joblib import load

rubrics = ['d_Prompt', 'd_Thesis', 'd_Claims', 'd_Evidence', 'd_Reasoning',
    'd_Organization', 'd_Rebuttal', 'd_Precision', 'd_Fluency', 'd_Coventions']
purposes = ['Organization', 'Word-Usage/Clarity', 'Claims/Ideas', 'Warrant/Reasoning/Backing',
    'Rebuttal/Reservation', 'Evidence', 'Precision', 'General Content Development']
sbert_cols = ['prev_para_embed_sbert', 'prev_sent_embed_sbert', 'target_embed_sbert',
    'fol_sent_embed_sbert', 'fol_para_embed_sbert']
closest_indices_cols = ['prev_para_closest_indices_sbert', 'prev_sent_closest_indices_sbert',
    'target_closest_indices_sbert', 'fol_sent_closest_indices_sbert', 'fol_para_closest_indices_sbert']
non_input = ['ID', 'Condition', 'draft', 'prev_para', 'prev_sent', 'target', 'fol_sent', 'fol_para',
    'prev_para_largest_weight_sbert', 'prev_sent_largest_weight_sbert', 'target_largest_weight_sbert',
    'fol_sent_largest_weight_sbert', 'fol_para_largest_weight_sbert']
random_forests = []
converters = {c: literal_eval for c in closest_indices_cols+sbert_cols}
edits = pd.read_csv('test_data/test_edits_sbert_embedding.csv', converters=converters)

concat_list = [edits]
for c in sbert_cols:
    #edits[c] = edits[c].str.strip('[]').str.strip(' ').str.replace('\n', '').str.replace('  ', ' ').str.split(' ')
    concat_list.append(pd.DataFrame(np.array(edits[c].tolist(), dtype=np.float32)))
edits = pd.concat(concat_list, axis=1)

out = []
for student in edits['ID'].unique():
    student_edits = edits[(edits['ID'] == student) & (edits['draft'] == 1)].reset_index(drop=True)
    #essay = pd.read_excel(f'Annotation_2018argrewrite_{student}.txt.xlsx', sheet_name='Old Draft')
    base = student_edits[student_edits['Add'] == 1].copy()
    mod = student_edits[student_edits['Add'] == 0].copy()
    delete = student_edits[student_edits['Add'] == 0].copy()
    mod['Modify'] = 1
    delete['Delete'] = 1
    base = pd.concat([base, mod, delete], axis=0, ignore_index=True)
    purposes_coded = []
    for p in purposes:
        tmp = base.copy()
        tmp[p] = 1
        purposes_coded.append(tmp)
    purposes_coded = pd.concat(purposes_coded, axis=0, ignore_index=True)
    test_X = purposes_coded.drop(non_input+closest_indices_cols+sbert_cols, axis=1)
    for r in rubrics:
        print(f'Predicting {r}')
        clf = load(f'models/{r}.joblib')
        purposes_coded[r] = clf.predict(test_X)
        purposes_coded[f'{r}_proba'] = clf.predict_proba(test_X)[:,1]
    out.append(purposes_coded)

out = pd.concat(out, axis=0, ignore_index=True)[non_input+['Add', 'Modify', 'Delete']+purposes+
    closest_indices_cols+sbert_cols+rubrics+[r+'_proba' for r in rubrics]]
out.to_csv('test_data/test_edits_sbert_prediction.csv', index=False)      
