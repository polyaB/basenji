## Manuscript models and data

[Cross-species regulatory sequence activity prediction. bioRxiv 6/2019.]()

*get_models.sh* - Download the saved TensorFlow models for human and mouse targets

Download the training/validation/test TFRecords	(319 Gb) from https://console.cloud.google.com/storage/browser/basenji_barnyard/data.

Scikit-learn random forest SNP classifiers for Mendelian disease and GWAS complex traits available from https://console.cloud.google.com/storage/browser/basenji_barnyard/sad/classifiers/.
Restore models using joblib.load.

All 1000 Genomes variant scores for human and mouse available from https://console.cloud.google.com/storage/browser/basenji_barnyard/sad/human/ and https://console.cloud.google.com/storage/browser/basenji_barnyard/sad/mouse/.

Autism variant scores available from https://console.cloud.google.com/storage/browser/basenji_barnyard/sad/autism/.
