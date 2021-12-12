from ast import literal_eval
from io import TextIOWrapper
import numpy as np
import pandas as pd
from joblib import load
import shap
from token_alignment import get_clause_from_bert_token_idx, SpecialTokenError

rubrics = ['d_Prompt', 'd_Thesis', 'd_Claims', 'd_Evidence', 'd_Reasoning', 'd_Organization',
    'd_Rebuttal', 'd_Precision', 'd_Fluency']
gaussian = ['d_Rebuttal', 'd_Fluency']
train_X = np.load('./numpy_data/sbert_train_X.npy')
train_X_gaussian = []
clf_gaussian = []
explainers = []
for r in rubrics:
    clf = load(f'models/{r}.joblib')
    if r in gaussian:
        train_y = np.load(f'./numpy_data/{r}_bin_train_y.npy')
        train_X_gaussian.append(train_X)
        clf_gaussian.append(clf)
        #samples = shap.sample(train_X, nsamples=50)
        #explainers.append(shap.KernelExplainer(clf.predict_proba, samples))
        explainers.append(None)
    else:
        explainers.append(shap.Explainer(clf))

closest_indices_cols = ['prev_para_closest_indices_sbert', 'prev_sent_closest_indices_sbert',
    'target_closest_indices_sbert', 'fol_sent_closest_indices_sbert', 'fol_para_closest_indices_sbert']
sbert_cols = ['prev_para_embed_sbert', 'prev_sent_embed_sbert', 'target_embed_sbert',
    'fol_sent_embed_sbert', 'fol_para_embed_sbert']
converters = {c: literal_eval for c in closest_indices_cols+sbert_cols}
edits = pd.read_csv('test_data/test_edits_sbert_prediction.csv', converters=converters)
edits = edits.fillna('')

types = ['Add', 'Modify', 'Delete']
purposes = ['Organization', 'Word-Usage/Clarity', 'Claims/Ideas', 'Warrant/Reasoning/Backing',
    'Rebuttal/Reservation', 'Evidence', 'Precision', 'General Content Development']
use_as_is_col = types+purposes
text_cols = ['prev_para', 'prev_sent', 'target', 'fol_sent', 'fol_para']

def get_reasons(r, row, within_target: int, others: int):
    print('Getting a reason for', r)
    embed_dim = 768
    data = row[use_as_is_col].to_numpy()
    for c in sbert_cols:
        data = np.append(data, np.array(row[c], dtype=np.float32))
    if r in gaussian:
        r_g_idx = gaussian.index(r)
        sample = train_X_gaussian[r_g_idx]
        for i in range(len(use_as_is_col)):
            sample_cand = sample[sample[:,i] == row[use_as_is_col[i]]]
            if len(sample_cand) == 0: break
            sample = sample_cand
        sample = np.median(sample, axis=0)
        exp = shap.KernelExplainer(clf_gaussian[r_g_idx].predict_proba, sample.reshape(1, len(sample)))
        shap_values = exp.shap_values(data)[1]
    else:
        exp = explainers[rubrics.index(r)]
        shap_values = exp(data.reshape(1, len(data)))[0].values
    if len(shap_values.shape) == 2:
      shap_values = shap_values[:, 1]
    shap_argsorted = np.argsort(shap_values)[::-1]
    target_texts = []
    others_texts = []
    skipped = False
    for arg in shap_argsorted:
        if len(target_texts) == within_target and len(others_texts) == others:
            break
        if arg < 11:
            skipped = True
            continue
        col = (arg - 11) // embed_dim
        if (col == 2 and len(target_texts) == within_target) or (col != 2 and len(others_texts) == others):
            continue
        idx = (arg - 11) % embed_dim
        token_idx = best_edit[closest_indices_cols[col]][idx]
        text = best_edit[text_cols[col]]
        try:
            clause = get_clause_from_bert_token_idx(text, token_idx, col < 2)
            if col == 2:
                target_texts.append(clause)
            else:
                others_texts.append(clause)
        except SpecialTokenError:
            continue

    return target_texts, others_texts, skipped
    

def write_feedback(writer: TextIOWrapper, edit_type, edit_purpose, r, target, row):
    if edit_type == 'Add':
        feedback_text = 'Add a new sentence here '
        if edit_purpose == 'Claims/Ideas':
            feedback_text += 'to introduce a new claim or idea. '
        elif edit_purpose == 'Warrant/Reasoning/Backing':
            feedback_text += 'to connect the previous and next sentences via reasoning. '
        elif edit_purpose == 'Rebuttal/Reservation':
            feedback_text += 'to strengthen rebuttal. '
        elif edit_purpose == 'Evidence':
            feedback_text += 'to support your thesis and/or claims with evidence. '
        elif edit_purpose == 'Precision':
            feedback_text += 'to be more precise. '
        else: # General Content Development
            feedback_text += 'to give more background information, contexts, etc. '

        if r == 'd_Prompt':
            _, others_texts, _ = get_reasons(r, row, within_target=0, others=1)
            feedback_text += f'You wrote "{others_texts[0]}", but this may not answer all parts of the prompt.'
        elif r == 'd_Thesis':
            feedback_text += 'You may be missing a thesis statement or your statement may be difficult to locate.'
        elif r == 'd_Claims':
            feedback_text += 'You may have zero or only one claim to support your thesis. Or your claims may be difficult to locate.'
        elif r == 'd_Evidence':
            _, others_texts, _ = get_reasons(r, row, within_target=0, others=1)
            feedback_text += f'Your claim "{others_texts[0]}" may miss supporting evidence.'
        elif r == 'd_Reasoning':
            _, others_texts, _ = get_reasons(r, row, within_target=0, others=2)
            feedback_text += f'You may lack connection between your ideas that "{others_texts[0]}" and "{others_texts[1]}"'
        elif r == 'd_Organization':
            _, others_texts, skipped = get_reasons(r, row, within_target=0, others=2)
            if skipped or others_texts[0] == others_texts[1]:
                feedback_text += 'You may not have a distinct introduction, body and conclusion.'
            else:
                feedback_text += f'The sequence of your ideas that "{others_texts[0]}" and "{others_texts[1]}" may be difficult to follow'
        elif r == 'd_Rebuttal':
            _, others_texts, skipped = get_reasons(r, row, within_target=0, others=1)
            if skipped:
                feedback_text += 'You may not have a rebuttal.'
            else:
                feedback_text += f'You may have to explain why the view that "{others_texts[0]}" exists and is incorrect.'
        elif r == 'd_Precision':
            _, others_texts, _ = get_reasons(r, row, within_target=0, others=1)
            feedback_text += f'You wrote "{others_texts[0]}", but it may need further clarification.'
        else: #'d_Fluency'
            _, others_texts, _ = get_reasons(r, row, within_target=0, others=1)
            feedback_text += f'You wrote "{others_texts[0]}", but it may contain words or phrases that you should explain more.'
        
        writer.write(f'<wow-tooltip class="tooltip"><span class="tooltip__label" aria-describedby="tooltip-demo-content" data-tooltip-placeholder>\
{target}</span><span class="tooltip-dropdown" data-tooltip-dropdown>\
<span role="tooltip" class="tooltip-dropdown__content">{feedback_text}</span></span></wow-tooltip> ')

    elif edit_type == 'Modify':
        feedback_text = 'Modify this sentence '

        if edit_purpose == 'Organization':
            feedback_text += 'for better organization. '
        elif edit_purpose == 'Word-Usage/Clarity':
            feedback_text += 'to clarify the meaning of this sentence. '
        elif edit_purpose == 'Claims/Ideas':
            feedback_text += 'to clarify your claim. '
        elif edit_purpose == 'Warrant/Reasoning/Backing':
            feedback_text += 'to make the connection between the previous and next sentences clearer. '
        elif edit_purpose == 'Rebuttal/Reservation':
            feedback_text += 'to strengthen rebuttal. '
        elif edit_purpose == 'Evidence':
            feedback_text += 'to strengthen the support to your thesis and/or claims. '
        elif edit_purpose == 'Precision':
            feedback_text += 'to make this sentence more precise. '
        else: # General Content Development
            feedback_text += 'to give enough background information, contexts, etc. '

        if r == 'd_Prompt':
            target_texts, _, _ = get_reasons(r, row, within_target=1, others=0)
            feedback_text += f'You wrote "{target_texts[0]}", but it may be off-topic.'
        elif r == 'd_Thesis':
            target_texts, _, _ = get_reasons(r, row, within_target=1, others=0)
            feedback_text += f'Your thesis "{target_texts[0]}" may be unclear or not indicate the stance you are taking toward the topic.'
        elif r == 'd_Claims':
            target_texts, others_texts, skipped = get_reasons(r, best_edit, within_target=1, others=1)
            if skipped:
                feedback_text += f'Your claim "{target_texts[0]}" may not align with your thesis "{others_texts[0]}".'
            else:
                feedback_text += f'Your claim "{target_texts[0]}" may be unclear.'
        elif r == 'd_Evidence':
            target_texts, others_texts, _ = get_reasons(r, row, within_target=1, others=1)
            feedback_text += f'The evidence "{target_texts[0]}" may not support your claim "{others_texts[0]}"'
        elif r == 'd_Reasoning':
            target_texts, others_texts, skipped = get_reasons(r, row, within_target=1, others=2)
            if skipped or others_texts[0] == others_texts[1]:
                feedback_text += f'Your reasoning "{target_texts[0]}" may just be a repeat of your claim "{others_texts[0]}".'
            else:
                feedback_text += f'Your reasoning "{target_texts[0]}" may not connect your ideas that "{others_texts[0]}" and "{others_texts[1]}"'
        elif r == 'd_Organization':
            target_texts, _, _ = get_reasons(r, row, within_target=1, others=0)
            feedback_text += f'You wrote "{target_texts[0]}", but it may not be related to this paragraph.'
        elif r == 'd_Rebuttal':
            target_texts, others_texts, _ = get_reasons(r, row, within_target=1, others=1)
            feedback_text += f'You wrote "{target_texts[0]}", but it does not explain why the view that "{others_texts[0]}" exists or is incorrect.'
        elif r == 'd_Precision':
            target_texts, _, _ = get_reasons(r, row, within_target=1, others=0)
            feedback_text += f'You wrote "{target_texts[0]}", but it may be too vague or general.'
        else: #'d_Fluency'
            target_texts, _, _ = get_reasons(r, row, within_target=1, others=0)
            feedback_text += f'You wrote “{target_texts[0]}”, but it may contain inappropriate word choices or sentence structures.'

        writer.write(f'<wow-tooltip class="tooltip"><span class="tooltip__label" aria-describedby="tooltip-demo-content" data-tooltip-placeholder>\
<mark>{target}</mark></span><span class="tooltip-dropdown" data-tooltip-dropdown>\
<span role="tooltip" class="tooltip-dropdown__content">{feedback_text}</span></span></wow-tooltip> ')

    else: # Delete
        feedback_text = 'Delete this sentence '
        if edit_purpose == 'Claims/Ideas':
            feedback_text += 'to clarify your claim. '
        elif edit_purpose == 'Warrant/Reasoning/Backing':
            feedback_text += 'to make the connection between the previous and next sentences smoother. '
        elif edit_purpose == 'Rebuttal/Reservation':
            feedback_text += 'to strengthen rebuttal. '
        elif edit_purpose == 'Evidence':
            feedback_text += 'to strengthen the support to your thesis and/or claims. '
        elif edit_purpose == 'Precision':
            feedback_text += 'to be more precise. '
        else: # General Content Development
            feedback_text += 'to avoid going off-topic. '

        if r == 'd_Prompt':
            target_texts, _, _ = get_reasons(r, row, within_target=1, others=0)
            feedback_text += f'You wrote "{target_texts[0]}", but it may be off-topic.'
        elif r == 'd_Thesis':
            target_texts, _, _ = get_reasons(r, row, within_target=1, others=0)
            feedback_text += f'Your thesis "{target_texts[0]}" may be unclear or not indicate the stance you are taking toward the topic.'
        elif r == 'd_Claims':
            target_texts, others_texts, _ = get_reasons(r, row, within_target=1, others=1)
            feedback_text += f'Your claim "{target_texts[0]}" may not align with your thesis "{others_texts[0]}"'
        elif r == 'd_Evidence':
            target_texts, others_texts, _ = get_reasons(r, row, within_target=1, others=1)
            feedback_text += f'The evidence "{target_texts[0]}" may not support your claim "{others_texts[0]}".'
        elif r == 'd_Reasoning':
            target_texts, others_texts, skipped = get_reasons(r, row, within_target=1, others=2)
            if skipped or others_texts[0] == others_texts[1]:
                feedback_text += f'Your reasoning "{target_texts[0]}" may just be a repeat of your claim "{others_texts[0]}"'
            else:
                feedback_text += f'Your reasoning "{target_texts[0]}" may not connect your ideas that "{others_texts[0]}" and "{others_texts[1]}"'
        elif r == 'd_Organization':
            target_texts, _, _ = get_reasons(r, row, within_target=1, others=0)
            feedback_text += f'You wrote "{target_texts[0]}", but it should not be in this paragraph.'
        elif r == 'd_Rebuttal':
            target_texts, others_texts, _ = get_reasons(r, row, within_target=1, others=1)
            feedback_text += f'You wrote "{target_texts[0]}", but it does not explain why the view that "{others_texts[0]}" exists or is incorrect.'
        elif r == 'd_Precision':
            target_texts, _, _ = get_reasons(r, row, within_target=1, others=0)
            feedback_text += f'You wrote "{target_texts[0]}", but it may be too vague or general.'
        else: #'d_Fluency'
            target_texts, _, _ = get_reasons(r, row, within_target=1, others=0)
            feedback_text += f'You wrote "{target_texts[0]}", but it may contain inappropriate word choices or sentence structures.'

        writer.write(f'<wow-tooltip class="tooltip"><span class="tooltip__label" aria-describedby="tooltip-demo-content" data-tooltip-placeholder>\
<del>{target}</del></span><span class="tooltip-dropdown" data-tooltip-dropdown>\
<span role="tooltip" class="tooltip-dropdown__content">{feedback_text}</span></span></wow-tooltip> ')
    print('Wrote feedback on', r)

def get_feedback(student: int, prev_para: str, prev_sent: str, target: str, fol_sent: str, fol_para: str):
    cur_edits = edits[(edits['ID'] == student) & (edits['prev_para'] == prev_para) & (
        edits['prev_sent'] == prev_sent) & (edits['target'] == target) & (edits['fol_sent'] == fol_sent) & (
            edits['fol_para'] == fol_para)]
    probabilities = cur_edits[[r + '_proba' for r in rubrics]].values
    arg_max_idx = np.dstack(np.unravel_index(np.argsort(probabilities.ravel()), probabilities.shape))[0][::-1]
    for max_idx in arg_max_idx:
        best_row = cur_edits.iloc[max_idx[0]]
        #if best_row[rubrics[max_idx[1]]] != 1:
        if best_row[rubrics[max_idx[1]] + '_proba'] < 0.6:
            return None, None, None, None
        if types[np.argmax(best_row[types].values)] != 'Modify':
            if purposes in ['Organization', 'Word-Usage/Clarity', 'Precision']:
                continue
        return types[np.argmax(best_row[types].values)], purposes[np.argmax(best_row[purposes].values)], rubrics[max_idx[1]], best_row
    return None, None, None, None

for student in edits['ID'].unique():
    print(f'Generating feedback for student {student}')
    essay = pd.read_excel(f'./test_data/Annotation_2018argrewrite_{student}.txt.xlsx', sheet_name='Old Draft')
    with open(f'./test_data/feedback_{student}.html', 'w') as f:
        f.write('<!DOCTYPE html>\n<html>\n<head>\n<link rel="stylesheet" href="tooltip-web-component.css">\n<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />\n</head>\n<body>\n')
        cur_para = 0
        for index, row in essay.iterrows():
            if cur_para != row['Original Paragraph No']: # The beginning of a paragraph
                if cur_para != 0:
                    f.write('</p><br>\n')
                # Addition between cur_para and the next paragraph
                prev_para = essay[essay['Original Paragraph No'] <= cur_para]['Sentence Content'].str.cat(sep=' ')
                fol_para = essay[essay['Original Paragraph No'] > cur_para]['Sentence Content'].str.cat(sep=' ')
                edit_type, edit_purpose, r, best_edit = get_feedback(student, prev_para, '', '', '', fol_para)
                if edit_type is not None: # Must be addition
                    f.write('<p>')
                    write_feedback(f, edit_type, edit_purpose, r, '&#10133;', best_edit)
                    f.write('</p><br>\n')

                f.write('<p>') # Start a new paragraph

                # Addition at the beginning of the next paragraph
                fol_sent = essay[essay['Original Paragraph No'] == cur_para+1]['Sentence Content'].str.cat(sep=' ')
                fol_para = essay[essay['Original Paragraph No'] > cur_para+1]['Sentence Content'].str.cat(sep=' ')
                edit_type, edit_purpose, r, best_edit = get_feedback(student, prev_para, '', '', fol_sent, fol_para)
                if edit_type is not None: # Must be addition
                    write_feedback(f, edit_type, edit_purpose, r, '&#10133;', best_edit)
                    
                cur_para += 1
            
            # Modification or deletion of the current row
            prev_para = essay[essay['Original Paragraph No'] < cur_para]['Sentence Content'].str.cat(sep=' ')
            prev_sent = essay[(essay['Original Paragraph No'] == cur_para) & (
                essay['Sentence Index'] < row['Sentence Index'])]['Sentence Content'].str.cat(sep=' ')
            fol_sent = essay[(essay['Original Paragraph No'] == cur_para) & (
                essay['Sentence Index'] > row['Sentence Index'])]['Sentence Content'].str.cat(sep=' ')
            fol_para = essay[essay['Original Paragraph No'] > cur_para]['Sentence Content'].str.cat(sep=' ')
            edit_type, edit_purpose, r, best_edit = get_feedback(student, prev_para, prev_sent, row['Sentence Content'], fol_sent, fol_para)
            if edit_type is not None:
                write_feedback(f, edit_type, edit_purpose, r, row['Sentence Content'], best_edit)
            else:
                f.write(row['Sentence Content'] + ' ')
            # Addition after the current row
            edit_type, edit_purpose, r, best_edit = get_feedback(student, prev_para, f"{prev_sent} {row['Sentence Content']}", '', fol_sent, fol_para)
            if edit_type is not None: # Must be addition
                write_feedback(f, edit_type, edit_purpose, r, '&#10133;', best_edit)

        # Addition after the last paragraph
        prev_para = essay['Sentence Content'].str.cat(sep=' ')
        edit_type, edit_purpose, r, best_edit = get_feedback(student, prev_para, '', '', '', '')
        if edit_type is not None: # Must be addition
            f.write('</p><br>\n<p>')
            write_feedback(f, edit_type, edit_purpose, r, '&#10133;', best_edit)

        f.write('</p><br>\n<script type="text/javascript" src="tooltip_position.js"></script>\n</body>\n</html>')
