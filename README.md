# Movie Genre Classification - Machine Learning INT3405E 20
This project showcases our commitment to the final Machine Learning course project, particularly in the realm of multilabel classification tasks.

## Content
- [Clone the repository](#step-1-clone-the-repository)
- [Install the dependencies](#step-2-install-the-dependencies)
- [Prepare the data](#step-3-prepare-the-data)
- [Create the paths](#step-4-create-the-paths)
- [Preprocess the data](#step-5-preprocess-the-data)
- [Train the model](#step-6-train-the-model)
- [Members](#members)

## Step 1. Clone the repository.
```
!git clone https://github.com/UwU-tao/csgolol.git
```

## Step 2. Install the dependencies.
```
!pip install -q torchmetrics gdown
```

## Step 3. Prepare the data.
``` 
!gdown 1hUqu1mbFeTEfBvl-7fc56fHFfCSzIktD
!unzip -qq ml1m.zip -d ml1m
```

## Step 4. Create the paths.
```
!mkdir /kaggle/working/csgolol/data
!mkdir /kaggle/working/csgolol/pretrained_models
!cp -r /kaggle/working/ml1m/content/dataset /kaggle/working/csgolol/data
``` 

## Step 5. Preprocess the data.
```
%cd /kaggle/working/csgolol
!python preprocess.py
```

## Step 6. Train the model.
```
!python main.py --model RatingwVGGnBERT_concat --num_epochs 25
```

## Members
Ngo Thuong Hieu - 21021491\
Nguyen Duc Hai - 21020623\
Pham Anh Duc - 21020187