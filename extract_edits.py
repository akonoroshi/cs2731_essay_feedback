import csv
import pandas as pd

def get_out_row(row, before, after, sent, prev_para, prev_sent, fol_sent, fol_para):
    if before < after:
        add = int(str(sent['Revision Operation Level 0']) == 'Add')
        delete = int(str(sent['Revision Operation Level 0']) == 'Delete')
        target = '' if str(sent['Revision Operation Level 0']) == 'Add' else sent['Sentence Content']
    else: # Inverse edits
        delete = int(str(sent['Revision Operation Level 0']) == 'Add')
        add = int(str(sent['Revision Operation Level 0']) == 'Delete')
        target = '' if str(sent['Revision Operation Level 0']) == 'Delete' else sent['Sentence Content']
    return [row['ID'],
            row['Condition'],
            before + after,
            add,
            int(str(sent['Revision Operation Level 0']) == 'Modify'),
            delete,
            1 if str(sent['Revision Purpose Level 0']) == 'Organization' else 0,
            1 if str(sent['Revision Purpose Level 0']) == 'Conventions/Grammar/Spelling' else 0,
            1 if str(sent['Revision Purpose Level 0']) == 'Word-Usage/Clarity' else 0,
            1 if str(sent['Revision Purpose Level 0']) == 'Claims/Ideas' else 0,
            1 if str(sent['Revision Purpose Level 0']) == 'Warrant/Reasoning/Backing' else 0,
            1 if str(sent['Revision Purpose Level 0']) == 'Rebuttal/Reservation' else 0,
            1 if str(sent['Revision Purpose Level 0']) == 'Evidence' else 0,
            1 if str(sent['Revision Purpose Level 0']) == 'Precision' else 0,
            1 if str(sent['Revision Purpose Level 0']) == 'General Content Development' else 0,
            prev_para,
            prev_sent,
            target,
            fol_sent,
            fol_para,
            row[f'Prompt_d{after}'] - row[f'Prompt_d{before}'],
            row[f'Thesis_d{after}'] - row[f'Thesis_d{before}'],
            row[f'Claims_d{after}'] - row[f'Claims_d{before}'],
            row[f'Evidence_d{after}'] - row[f'Evidence_d{before}'],
            row[f'Reasoning_d{after}'] - row[f'Reasoning_d{before}'],
            row[f'Organization_d{after}'] - row[f'Organization_d{before}'],
            row[f'Rebuttal_d{after}'] - row[f'Rebuttal_d{before}'],
            row[f'Precision_d{after}'] - row[f'Precision_d{before}'],
            row[f'Fluency_d{after}'] - row[f'Fluency_d{before}'],
            row[f'Coventions_d{after}'] - row[f'Coventions_d{before}']
            ]

def write_edits(csvwriter, row, before, after, df_old, df_new):
    if before < after:
        old_p_num = 'Original Paragraph No'
        new_p_num = 'New Paragraph No'
    else:
        new_p_num = 'Original Paragraph No'
        old_p_num = 'New Paragraph No'
    
    for _, sent in df_old.iterrows():
        if not pd.isna(sent['Revision Operation Level 0']): # Modify or Delete
            prev_para = df_old[df_old[old_p_num] < sent[old_p_num]][
                'Sentence Content'].str.cat(sep=' ')
            prev_sent = df_old[(df_old[old_p_num] == sent[old_p_num]) & (
                df_old['Sentence Index'] < sent['Sentence Index'])]['Sentence Content'].str.cat(sep=' ')
            fol_sent = df_old[(df_old[old_p_num] == sent[old_p_num]) & (
                df_old['Sentence Index'] > sent['Sentence Index'])]['Sentence Content'].str.cat(sep=' ')
            fol_para = df_old[df_old[old_p_num] > sent[old_p_num]][
                'Sentence Content'].str.cat(sep=' ')
            csvwriter.writerow(get_out_row(row, before, after, sent, prev_para, prev_sent, fol_sent, fol_para))

    for _, sent in df_new.iterrows():
        if (before < after and str(sent['Revision Operation Level 0']) == 'Add') or\
            (before > after and str(sent['Revision Operation Level 0']) == 'Delete'): # Add
            prev_row = None
            for i in range(int(sent['Sentence Index']) - 1, 0, -1):
                if (before < after and df_new[df_new['Sentence Index'] == i].iloc[0]['Aligned Index'] != 'ADD') or\
                    (before > after and df_new[df_new['Sentence Index'] == i].iloc[0]['Aligned Index'] != 'DELETE'):
                    prev_row = df_new[df_new['Sentence Index'] == i].iloc[0]
                    break
            fol_row = None
            for i in range(int(sent['Sentence Index']) + 1, len(df_new) + 1):
                if (before < after and df_new[df_new['Sentence Index'] == i].iloc[0]['Aligned Index'] != 'ADD') or\
                    (before > after and df_new[df_new['Sentence Index'] == i].iloc[0]['Aligned Index'] != 'DELETE'):
                    fol_row = df_new[df_new['Sentence Index'] == i].iloc[0]
                    break
                    
            if prev_row is None and fol_row is None: # No sentence from the previous draft
                break
            if fol_row is None:
                prev_old_sent = df_old[df_old['Sentence Index'] == int(prev_row['Aligned Index'].split(',')[0])].iloc[0]
                if sent[new_p_num] == prev_row[new_p_num]: # Added at the end of the paragraph
                    prev_para = df_old[df_old[old_p_num] < prev_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                    prev_sent = df_old[df_old[old_p_num] == prev_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                    fol_sent = ''
                    fol_para = df_old[df_old[old_p_num] > prev_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                else: # Added at the beginning of the paragraph or a new paragraph was added at the end
                    prev_para = df_old[df_old[old_p_num] <= prev_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                    prev_sent = ''
                    fol_sent = df_old[df_old[old_p_num] == prev_old_sent[old_p_num] + 1][
                        'Sentence Content'].str.cat(sep=' ')
                    fol_para = df_old[df_old[old_p_num] > prev_old_sent[old_p_num] + 1][
                        'Sentence Content'].str.cat(sep=' ')
                    
            elif prev_row is None:
                fol_old_sent = df_old[df_old['Sentence Index'] == int(fol_row['Aligned Index'].split(',')[0])].iloc[0]
                if sent[new_p_num] == fol_row[new_p_num]: # Added at the beginning of the paragraph
                    prev_para = df_old[df_old[old_p_num] < fol_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                    prev_sent = ''
                    fol_sent = df_old[df_old[old_p_num] == fol_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                    fol_para = df_old[df_old[old_p_num] > fol_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                else: # Added at the end of the paragraph or a new paragraph was added at the beginning
                    prev_para = df_old[df_old[old_p_num] < fol_old_sent[old_p_num] - 1][
                        'Sentence Content'].str.cat(sep=' ')
                    prev_sent = df_old[df_old[old_p_num] == fol_old_sent[old_p_num] - 1][
                        'Sentence Content'].str.cat(sep=' ')
                    fol_sent = ''
                    fol_para = df_old[df_old[old_p_num] >= fol_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                    
            else:
                prev_old_sent = df_old[df_old['Sentence Index'] == int(prev_row['Aligned Index'].split(',')[0])].iloc[0]
                fol_old_sent = df_old[df_old['Sentence Index'] == int(fol_row['Aligned Index'].split(',')[0])].iloc[0]
                if sent[new_p_num] == prev_row[new_p_num] and\
                    sent[new_p_num] == fol_row[new_p_num]: # Added in the middle of the paragraph
                    prev_para = df_old[df_old[old_p_num] < prev_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                    prev_sent = df_old[(df_old[old_p_num] == prev_old_sent[old_p_num]) & (
                        df_old['Sentence Index'] < fol_old_sent['Sentence Index'])]['Sentence Content'].str.cat(sep=' ')
                    fol_sent = df_old[(df_old[old_p_num] == fol_old_sent[old_p_num]) & (
                        df_old['Sentence Index'] >= fol_old_sent['Sentence Index'])]['Sentence Content'].str.cat(sep=' ')
                    fol_para = df_old[df_old[old_p_num] > fol_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                elif sent[new_p_num] == prev_row[new_p_num]: # Added at the end of the paragraph
                    prev_para = df_old[df_old[old_p_num] < prev_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                    prev_sent = df_old[df_old[old_p_num] == prev_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                    fol_sent = ''
                    fol_para = df_old[df_old[old_p_num] >= fol_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                elif sent[new_p_num] == fol_row[new_p_num]: # Added at the beginning of the paragraph
                    prev_para = df_old[df_old[old_p_num] <= prev_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                    prev_sent = ''
                    fol_sent = df_old[df_old[old_p_num] == fol_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                    fol_para = df_old[df_old[old_p_num] > fol_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                else: # A new paragraph was added
                    prev_para = df_old[df_old[old_p_num] <= prev_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                    prev_sent = ''
                    fol_sent = ''
                    fol_para = df_old[df_old[old_p_num] >= fol_old_sent[old_p_num]][
                        'Sentence Content'].str.cat(sep=' ')
                    
            csvwriter.writerow(get_out_row(row, before, after, sent, prev_para, prev_sent, fol_sent, fol_para))

scores = pd.read_excel('Updated-REVISION Revision-Scores-Survey per Student.xlsx', sheet_name='Data')
out_header = ['ID', 'Condition', 'revision', 'Add', 'Modify', 'Delete', 'Organization', 'Conventions/Grammar/Spelling',
    'Word-Usage/Clarity', 'Claims/Ideas', 'Warrant/Reasoning/Backing', 'Rebuttal/Reservation', 'Evidence', 'Precision',
    'General Content Development', 'prev_para', 'prev_sent', 'target', 'fol_sent', 'fol_para', 'd_Prompt', 'd_Thesis', 'd_Claims',
    'd_Evidence', 'd_Reasoning', 'd_Organization', 'd_Rebuttal', 'd_Precision', 'd_Fluency', 'd_Coventions']

with open('all_edits.csv', 'w') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(out_header)
    for _, row in scores.iterrows():
        for revision in ('12', '23'):
            path = f"Condition{row['Condition']}/Rev{revision} annotated/Annotation_2018argrewrite_{row['ID']}.txt.xlsx"
            while True:
                try:
                    old = pd.read_excel(path, sheet_name='Old Draft')
                    new = pd.read_excel(path, sheet_name='New Draft')
                except FileNotFoundError:
                    path_split = path.split('.', 1)
                    path = path_split[0] + '_NEW.' + path_split[1]
                else:
                    break

            write_edits(csvwriter, row, revision[0], revision[1], old, new)
            write_edits(csvwriter, row, revision[1], revision[0], new, old)
            