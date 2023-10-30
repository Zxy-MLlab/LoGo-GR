# LoGo-GR
Local to global graphical reasoning framework for extracting structured information from biomedical literature

# requirement
```python 
python==3.8 

torch==1.13.1 

transformers==3.0.0 

numpy==1.19.5

scikit-learn==0.23.1

stanfordcorenlp

```

# Dataload
Download DV dataset from URL: xxx

Download CDR dataset from URL: [](https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip)https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip, 

Download GDA dataset from URL: [](https://bitbucket.org/alexwuhkucs/gda-extraction/get/fd4a7409365e.zip)https://bitbucket.org/alexwuhkucs/gda-extraction/get/fd4a7409365e.zip,

Download DocRED dataset from URL: [](https://github.com/thunlp/DocRED)https://github.com/thunlp/DocRED ,

and place the file in the data folder.

# Pretrain Language model
Download scibert or bert models from URL: [](https://huggingface.co/allenai/scibert_scivocab_uncased)https://huggingface.co/allenai/scibert_scivocab_uncased or xxx.

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

# Runnning

Run the main.py file to train and test the model:
```python
python main.py
```
