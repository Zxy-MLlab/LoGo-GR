# LoGo-GR
Local to global graphical reasoning framework for extracting structured information from biomedical literature

# Requirement
```python 
python==3.8 

torch==1.13.1 

transformers==3.0.0 

numpy==1.19.5

scikit-learn==0.23.1

stanfordcorenlp

```

# Dataload
Download DV dataset reference: Cross-Sentence N-ary Relation Extraction with Graph LSTMs,

Download CDR dataset from URL: [](https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip)https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip, 

Download GDA dataset from URL: [](https://bitbucket.org/alexwuhkucs/gda-extraction/get/fd4a7409365e.zip)https://bitbucket.org/alexwuhkucs/gda-extraction/get/fd4a7409365e.zip,

Download DocRED dataset from URL: [](https://github.com/thunlp/DocRED)https://github.com/thunlp/DocRED .

Please extract the downloaded dataset file and place it in the ./data folder, as shown below:

For CDR dataset:
```
--data
  --CDR_data
    --CDR_DevelopmentSet.PubTator.txt
    --CDR_TrainingSet.PubTator.txt
    --CDR_TestSet.PubTator.txt
```

# Pretrain Language model
Download scibert or bert models from URL: [](https://huggingface.co/allenai/scibert_scivocab_uncased)https://huggingface.co/allenai/scibert_scivocab_uncased or [](https://huggingface.co/bert-base-uncased)https://huggingface.co/bert-base-uncased.

# Preprocessing data
To process the dataset, execute the following code:

preprocessing DV dataset:
```python
python dv_preprocessing.py
```
or preprocessing CDR dataset:
```python
python cdr_preprocessing.py
```
or preprocessing GDA dataset:
```python
python gda_preprocessing.py
```
or preprocessing DocRED dataset:
```python
python doc_preprocessing.py
```

When the program is finished, the following files are written to the ./prepro_data folder:
```
--CDR_DevelopmentSet.PubTator.json
--CDR_TrainingSet.PubTator.json
--CDR_TestSet.PubTator.json
```

# Runnning

Run the main.py file to train and test the model:
```python
python main.py
```

# Results
```
...
BEST: Epoch: 36 | NT F1: 0.7071072883657764 | F1: 0.7071072883657764 | Intra F1: 0.7372654155495979 | Inter F1: 0.6424581005586593 | Precision: 0.696078431372549 | Recall: 0.718491260349586 | AUC: 0.6109029466769528 | THETA: 0.9996838569641113
```
