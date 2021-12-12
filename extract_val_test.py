import csv
import pandas as pd

def get_out_row(s_id, cond, add, prev_para, prev_sent, target, fol_sent, fol_para, is_new):
    return [s_id,
            cond,
            int(is_new) + 1,
            add,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            prev_para,
            prev_sent,
            target,
            fol_sent,
            fol_para,
            ]

def write_possible_edits(csvwriter, s_id, cond, df, is_new):
    if is_new:
        para_col = 'New Paragraph No'
    else:
        para_col = 'Original Paragraph No'
    
    for _, sent in df.iterrows():
        # Modify or Delete
        prev_para = df[df[para_col] < sent[para_col]]['Sentence Content'].str.cat(sep=' ')
        prev_sent = df[(df[para_col] == sent[para_col]) & (
            df['Sentence Index'] < sent['Sentence Index'])]['Sentence Content'].str.cat(sep=' ')
        fol_sent = df[(df[para_col] == sent[para_col]) & (
            df['Sentence Index'] > sent['Sentence Index'])]['Sentence Content'].str.cat(sep=' ')
        fol_para = df[df[para_col] > sent[para_col]]['Sentence Content'].str.cat(sep=' ')
        csvwriter.writerow(get_out_row(
            s_id, cond, 0, prev_para, prev_sent, sent['Sentence Content'], fol_sent, fol_para, is_new))

        # Addition between sentences or at the beginning of a paragraph
        fol_sent = df[(df[para_col] == sent[para_col]) & (
            df['Sentence Index'] >= sent['Sentence Index'])]['Sentence Content'].str.cat(sep=' ')
        csvwriter.writerow(get_out_row(s_id, cond, 1, prev_para, prev_sent, '', fol_sent, fol_para, is_new))

    # Addition between paragraphs or at the end of a paragraph
    for p in range(int(max(df[para_col])) + 1):
        prev_para = df[df[para_col] <= p]['Sentence Content'].str.cat(sep=' ')
        fol_para = df[df[para_col] > p]['Sentence Content'].str.cat(sep=' ')
        csvwriter.writerow(get_out_row(s_id, cond, 1, prev_para, '', '', '', fol_para, is_new))

        if p != 0:
            prev_para = df[df[para_col] < p]['Sentence Content'].str.cat(sep=' ')
            prev_sent = df[df[para_col] == p]['Sentence Content'].str.cat(sep=' ')
            fol_para = df[df[para_col] > p]['Sentence Content'].str.cat(sep=' ')
            csvwriter.writerow(get_out_row(s_id, cond, 1, prev_para, prev_sent, '', '', fol_para, is_new))


out_header = ['ID', 'Condition', 'draft', 'Add', 'Modify', 'Delete', 'Organization', 'Word-Usage/Clarity', 
    'Claims/Ideas', 'Warrant/Reasoning/Backing', 'Rebuttal/Reservation', 'Evidence', 'Precision',
    'General Content Development', 'prev_para', 'prev_sent', 'target', 'fol_sent', 'fol_para']
val_id_cond = {14: 'C', 29: 'D', 37: 'B', 54: 'C', 62: 'C'}
test_id_cond = {36: 'A', 42: 'C', 45: 'B'}

with open('./val_data/val_edits.csv', 'w') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(out_header)
    for s_id, cond in val_id_cond.items():
        path = f'./val_data/Annotation_2018argrewrite_{s_id}.txt.xlsx'
        old = pd.read_excel(path, sheet_name='Old Draft')
        new = pd.read_excel(path, sheet_name='New Draft')

        write_possible_edits(csvwriter, s_id, cond, old, False)
        write_possible_edits(csvwriter, s_id, cond, new, True)

with open('./test_data/test_edits.csv', 'w') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(out_header)
    for s_id, cond in test_id_cond.items():
        path = f'./test_data/Annotation_2018argrewrite_{s_id}.txt.xlsx'
        old = pd.read_excel(path, sheet_name='Old Draft')
        new = pd.read_excel(path, sheet_name='New Draft')

        write_possible_edits(csvwriter, s_id, cond, old, False)
        write_possible_edits(csvwriter, s_id, cond, new, True)
            