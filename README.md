# Sarcasm Detection Model

## Graphical Model Training

generated corpus

1. `cd ./preprocess`
2. Run `python remove_words_all.py twcvd2`
3. Run `python build_graph_all.py twcvd2`
4. `cd ../`

then go back to root directory and run

`python main.py --model GAT`

## Analysis

### sarcasm_dataset_analysis.ipynb && sarcasm_dataset_analysis_2.ipynb

This notebook is about the analysis on the sarcasm dataset. You will need to open this notebook in Google Colab, upload the /dataset/sarcasm_dataset.jsonl as prompted, and run the file to get analysis result.

### covid_group_analysis.ipynb && temporal.ipynb

This notebook performs the analysis of sarcasm in misinformation and counter-misinformation tweets. You will need to open this notebook in Google Colab, upload the /dataset/covid_19/g5_labeled_Sarcasm.csv as prompted, and run the file to get analysis result.

