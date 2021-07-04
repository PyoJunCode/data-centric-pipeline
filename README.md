# Data-centric AI Comepetition



<p align="center"><img src="https://github.com/PyoJunCode/data-centric-pipeline/blob/master/images/logo.png"  style="zoom:50%;"></p>  



Competition : [Homepage](https://https-deeplearning-ai.github.io/data-centric-comp/?utm_source=thebatch&utm_medium=newsletter&utm_campaign=dc-ai-competition&utm_content=dl-ai)

Database : [DVC github](https://github.com/PyoJunCode/data-centric-AI-competition)

*21/06/19 ~ In progress*



<p align="center">tensorflow:2.5.0,	  TFX:0.30.0,	  	Kubeflow:1.6.1</p>

<br>

## Intro

실제 ML service의 Production 환경에서 오랜 시간 Model을 튜닝하여 1%의 정확도 향상을 얻게되는 반면, Data를 Cleansing하고 적절한 preprocessing을 거쳐 10%의 정확도 향상을 얻는 경우가 있는 만큼 Data Engineering의 중요도는 올라가고 있습니다.

<br>

평소에 접하는 Kaggle이나 Research 환경처럼 고정된 Dataset에 대해 모델을 Tuning해 나가는것이 아니라, 고정된 Model에 대해 Data를 가공하여 정확도를 높여나가는대회입니다. 


<br>
처음 지급되는 2000 + 800개 가량의 Hand-writing roman-number 이미지를 preprocessing, augmentation 과정을 거쳐 Resnet 50 기반의 Fixed 모델 에 학습시켜 최대한 높은 정확도를 얻어야 합니다.<br>

<br>
augmentation으로 늘릴 수 있는 Train, evaluation 데이터의 총 합이 10,000개로 제한되어 있기 때문에 무작정 데이터를 늘리는것은 효율적이지 못합니다.<br>

<br>그렇기 때문에 체계화된 **Error analysis 환경**을 만드는것이 관건이라고 생각했습니다.

<br>

고정된 Model에 대해서 Training - Error analysis - Add/fix data 의 반복적인 과정을 수행해야 하기 때문에효율적인 작업을 위해 기존과는 **다소 다른 Workflow를 가진 Pipeline**를 구축할 필요가 있었습니다.

<p align="center"><img src="https://github.com/PyoJunCode/data-centric-pipeline/blob/master/images/iteration.png" ></p>  



<br>

## Overall



GCP(Google Cloud Platform) 위에 **Kubeflow Pipeline**과 추가적으로 Local과 GCP를 연결하는 **Data ETL pipeline**를 구축하였습니다.

<p align="center"><img src="https://github.com/PyoJunCode/data-centric-pipeline/blob/master/images/structure.png" ></p>  

<br>전체 흐름을 요약하면 Data pipeline -> ML Pipeline -> Feedback -> Preprocessing/Augmentation -> 반복 의 Workflow를 가지고 있습니다.

**Index**

-> [AI platform](https://github.com/PyoJunCode/data-centric-pipeline#AI-platform)

-> [Kubeflow pipeline](https://github.com/PyoJunCode/data-centric-pipeline#Kubeflow-Pipeline)

-> [Data ETL pipeline](https://github.com/PyoJunCode/data-centric-pipeline#Data-ETL-Pipeline)

-> [Error Analysis](https://github.com/PyoJunCode/data-centric-pipeline#Error-Analysis)

-> [Outcome](https://github.com/PyoJunCode/data-centric-pipeline#Outcome)

-> [Components](https://github.com/PyoJunCode/data-centric-pipeline#Components)

<br>

## AI-platform

AI platform 내에 있는 Jupyterlab notebook을 통해 ML/Data pipeline을 생성하고, 통제합니다.



### control.ipynb 
ML pipeline launcher



```python
 #Create and run pipeline
!tfx run create --pipeline-name={PIPELINE_NAME} --endpoint={ENDPOINT}
```

```python
# Update the pipeline
!tfx pipeline update \
--pipeline-path=kubeflow_runner.py \
--endpoint={ENDPOINT}
# run the pipeline.
!tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT}
```

위의 명령어 셀을 실행하면 Kubernetes 클러스터의 Kubeflow에 Pipeline이 생성되고 실행됩니다.

**Output**

Kubeflow 대쉬보드에 가면 설정한 Pipeline이 생성되고 작동하는것을 확인할 수 있습니다.

<p align="center"><img src="https://github.com/PyoJunCode/data-centric-pipeline/blob/master/images/pipelines.png" style="zoom:50%;" ></p>  

<br>

### interactive.ipynb 
Data ETL pipeline launcher



AI platform은 DVC와 연결되어있습니다.

```
!git pull origin <data version branch>
!dvc pull
```

위의 셀을 실행하여 원하는 Data version을 checkout 하고 pull 할 수 있습니다.



TFX components의 특성 상 Data는 csv나 tfrecords 형식으로 제공되어야 하기 때문에 아래의 명령어 셀을 실행하여 폴더 별로 저장되어있는 Raw Image를 **.tfrecords로 변환**해 줍니다.

```python
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Convert Raw image to .tfrecords
def convert_to(images, label, path):
  
  labels = {1:"i", 2:"ii", 3:"iii", 4:"iv", 5:"v", 6:"vi", 7:"vii", 8:"viii", 9:"ix", 10:"x"}
  num_examples=len(images)

  filename = os.path.join(path+cnv+ labels[label] + '.tfrecords')
  print('Writing', filename)
  writer = tf.io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = open(path+f'{labels[label]}/'+images[index],'rb').read()
    image_shape = tf.io.decode_jpeg(image_raw).shape
    width = image_shape[0]
    height = image_shape[1]
    depth = image_shape[2]

    example = tf.train.Example(features=tf.train.Features(feature={
      
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
```


```python
#pass Data foler's path
def convert_to_tfrecords(target):
  for index in range(len(labels)):
      files = [name for name in os.listdir(target+labels[index])]
      if '.DS_Store' in files:
        files.remove('.DS_Store')
      print('target:',labels[index])
      print(files)
      convert_to(files,index+1,target)

convert_to_tfrecords(val_data_dir)
convert_to_tfrecords(train_data_dir)
```

이 작업은 Tensorflow에서 제공하는 'tf.keras.preprocessing.image_dataset_from_directory' 함수와 유사하게 동작하면서 tfrecords 파일을 폴더에 저장한다고 생각하면 됩니다.



마지막으로 아래의 명령어 쉘을 실행하여 Pipeline이 데이터를 불러올 **Cloud Store로 변환된 데이터를 복사**합니다.

```
!gsutil -m cp -r data/train/ gs://{GOOGLE_CLOUD_PROJECT}-kubeflowpipelines-default/data-centric/data/train/
!gsutil -m cp -r data/val/ gs://{GOOGLE_CLOUD_PROJECT}-kubeflowpipelines-default/data-centric/data/val/
```



<br>



## Kubeflow-Pipeline

<p align="center"><img src="https://github.com/PyoJunCode/data-centric-pipeline/blob/master/images/pipeline_structure.png" style="zoom:50%;" ></p>  



Kubernetes 클러스터의 Kubeflow pipeline의 DAG는 위와 같습니다.



- **ExampleGen** : .tfrecords 파일들을 읽어 [Train, Eval] example component입니다.

- **StatisticGen** : ExampleGen의 데이터에 대한 통계 정보를 갖는 component입니다.
- **SchemaGen** : ExampleGen의 Schema 정보를 갖는 component입니다.
- **ExampleValidator** : StatisticGen과 SchemaGen의 정보를 토대로 데이터의 Anomalies를 검사합니다.
- **Transform** : example에 정의된 preprocessing 과정을 적용합니다.
- **Trainer** : 정의된 Model을 example을 통해 Training 합니다. 
- **Evaluator** : Training이 끝난 Model의 결과 정보를 비교합니다.



Kubeflow - Run - Graph 에서 Pipeline의 각 Component가 갖고있는 정보를 Visualization하여 볼 수 있습니다.



**해당 프로젝트의 경우 Fixed된 모델에 대해 Data에만 변화를 주기 때문에 따로 모델의 정보를 저장하지않고, Pipeline에 Pusher components를 구성하지 않았습니다.** 

<br>

## Data-ETL-Pipeline

<p align="center"><img src="https://github.com/PyoJunCode/data-centric-pipeline/blob/master/images/datapipeline.png" ></p>  



Data Pipeline의 중심은 Google Cloud Storage입니다.



우선 로컬에서 전처리/증강 한 데이터들을 Git과 DVC에 push합니다.



Google Cloud Storage의 DVC bucket에 push된 Data의 cache 정보가 저장됩니다.<br>

<br>

[AI platform의 notebook](https://github.com/PyoJunCode/data-centric-pipeline#interactive.ipynb ) 에서 명령어 쉘을 실행하는 것으로 ETL Pipeline의 progress가 실행됩니다.<br>

- Cloud Storage의 Cache Bucket로부터 Raw Image data를 **Extract** 합니다.

- Raw Image Data를 .tfrecords로 **Transform** 하는 과정을 거칩니다.

- 마지막으로 transform된 data들을  Cloud Storage의 Data bucket에 **Load** 시켜 Pipeline에 활용될 수 있게 합니다.

<br>
Load된 Data를 가지고 Pipeline 과정들을 거치면서 Error validation을 진행한 뒤, 해당 정보를 feedback으로  다시 Local에서 전처리/증강 과정을 거쳐 일련의 과정을 반복 (Reiteration)합니다.



<br>



## Error-Analysis



Data가 여러개의 feature를 갖는것이 아니므로 Pipeline의 **StatistcsGen**과 **Evaluator**의 결과를 visualization하여 데이터 버젼 별로 label 통계와 학습 정확도를 비교분석 하였습니다.


StatisticsGen Visualization
<p align="center"><img src="https://github.com/PyoJunCode/data-centric-pipeline/blob/master/images/val1.png" style="zoom:50%;" ></p>  

<br>

Evaluator Visualization
<p align="center"><img src="https://github.com/PyoJunCode/data-centric-pipeline/blob/master/images/val2.png" style="zoom:50%;" ></p>  

<br>

Image recognition의 경우, **Human-level performance**라는 좋은 baseline이존재합니다. 따라서 validation set에 대해 오답이 나온 example 들을 manually하게 살펴보며 오류의 category를 정리했습니다.

<p align="center"><img src="https://github.com/PyoJunCode/data-centric-pipeline/blob/master/images/erranalysis.png" ></p>  



해당 오류 Category를 커버하기 위해서 최종적으로 Data를 Validating하는 과정은



**Preprocessing**

1. erosion->dilation
2. Adaptive Thresholding
3. Median filter

과정을 통해 Note의 줄, Noise, Blur 등을 제거하였고



**Augmentation**

1. Random rotate (+/- 20 degree)
2. Random zoom
3. Random Clip
4. Random Brightness

과정을 통해 부족한 Data를 증강하였습니다.





## Outcome



처음2000개의 Data에 대해서 몇번의 Preprocessing, augmentation을 거쳐서 현재 약 8천개 규모의 Data v3.0 을 만들었습니다.

Eval set에 대해서 기존 **68%** 에서 **88.7%** 로 향상되었고,

 대회 주최측에서 제공한 작은규모의 label book(test set)에 대해서는 기존 **55%** 에서 **100%** 의 정확도를 기록하였습니다.

**Initial Data**

<p align="center"><img src="https://github.com/PyoJunCode/data-centric-pipeline/blob/master/images/init.png" style="zoom:50%;" ></p>  

**Data v3.0**

<p align="center"><img src="https://github.com/PyoJunCode/data-centric-pipeline/blob/master/images/v3.png" style="zoom:50%;" ></p>  

(7/4 현재 leader board 기준 evaluation set 정확도 최상위권)





<br>

## Components

Repository의 구성은 아래와 같습니다.



- models
  - keras
    - model.py
  - preprocessing.py
- pipeline
  - configs.py
  - pipeline.py
- interactive.ipynb
- template.ipynb



model.py : 학습에 사용될 Model의 코드입니다. 

preprocessing.py : Pipeline의 transform 과정 중 사용하는 전처리 모듈입니다.



configs.py : pipeline의 batch size, #epoch 등의 정보가 들어있습니다.

pipeline.py : pipeline을 구성하는 모듈입니다.



interactive.ipynb : Pipeline을 구성할 경우, Model은 필수적으로 .tfrecord 형식으로 Data를 입력받아야합니다. 이를 위해 Data directory에 있는 이미지들을 .tfrecord로 바꿔주는 Utility 함수를 제공합니다.

또한 전체적인 Pipeline을 kubeflow dashboard가 아니라 notebook의 interactive context를 통해 실행해볼 수 있습니다.



template.ipynb : kubernates서버와 Cloud Storage 등 Pipeline을 구성하는데에 필요한 사전 연결 작업을 수행합니다.
