# cs2731_essay_feedback

This is the repository for the final project of CS 2731 at the University of Pittsburgh

## Prerequisites 

- ArgReWrite V2 corpus. It should be available to public soon, but at this moment, ask Tazin Afrin (taa74@pitt.edu) for the corpus. 
- packages in requirements.txt.

## How to run the codes

1. Once you have the corpus, save the folders named ConditionA to ConditionD at the same level as this file.
2. Create folders named "models", "numpy_data", and "val_data".
3. Copy the essays for validation and test to "val_data" and "test_data" respectively. The student id's for validation are 14, 29, 37, 54, and 62, and the id's for test are 36, 42, and 45. They were chosen randomly.
4. First, run `extract_edits.py` and `extract_val_test.py` to get `all_edits.csv`, `val_data/val_edits.csv`, and `test_data/test_edits.csv`.
5. Then, run [this colab notebook](https://colab.research.google.com/drive/12iw1PXlT5Ks_LZnwu-VhKS05iOMgkVqa?usp=sharing) to get sentence embedding for each of the csv file from the previous step. Save the output in the corresponding folders. 
6. Run scripts in the order of `save_to_npy.py`, `save_classifiers.py`, `predict_test.py`, and `generate_feedback_shap.py`.
7. The feedback is saved as html files in the `test_data` folder. You can open them in your favorite browser.
