# Kubeflow pipeline on GCP

ML pipeline using GCP-Kubernetes-Kubeflow

Pipeline for Data-centric AI competition. [Homepage](https://https-deeplearning-ai.github.io/data-centric-comp/?utm_source=thebatch&utm_medium=newsletter&utm_campaign=dc-ai-competition&utm_content=dl-ai)

Database : [DVC github](https://github.com/PyoJunCode/data-centric-AI-competition)

<br>

## Overall Structure

![overallstructure](https://user-images.githubusercontent.com/47979730/123545682-3a68a400-d794-11eb-9adb-8748aebaf373.PNG)

Pipeline의 전체적인 구조는 위와 같이 되어있습니다.



1. **AI platform**

   AI platform의 Jupyter lab notebook 상에서의 제어를 통해 Kubernetes 클러스터 안에 Kubeflow pipeline을 추가하고 Model의 Training에 쓸 Data를 변환하여 Cloud에 Serving합니다.

   notebook에는 Pipeline controller, Models, Data Converter 등의 파일들이 있습니다.



2. **Kubernetes cluster**

   Cloud platform 내의 kubernetes서버는 AI Platform으로 부터 전달된 Pipeline 명령어를 통해 Kubeflow Pipeline을 구성하고 task를 실행합니다. Kubeflow dashboard를 통해 Pipeline 각 과정의 결과물을 visualization하여 real-time으로 확인할 수 있습니다. *(Pipeline의 detail은 아래에 설명.)*

   학습을 하기 위한 Data를 Cloud Storage에 직접 접근하여 불러오고, Pipeline의 각 실행 과정 중 생성된 Artifact들또한 Cloud Storage에 저장합니다. 



3. **Cloud Storage**

   Training data, Pipeline Artifacts 등을 보관합니다. Training Data의 경우 Local computer와 AI Platform 사이에서 DVC 를 사용하여 상호적으로 Push/Pull 을 통해 관리됩니다.

   

4. **Local**

   Pipeline의 Model evaluator의 결과를 분석하고 해당 결과를 참고하여 다시 새로운 Data를 augment합니다.

   Data의 용량과 파일개수가 매우 크므로 Data는 기본적으로 DVC에 의해 직접 Google Cloud Storage와 연동되어 push/pull 합니다. 또한 각 Data의 version 정보는 따로 Git에 push하여 관리합니다.

<br>

## Pipeline



![pipelinestructure](https://user-images.githubusercontent.com/47979730/123546350-30947000-d797-11eb-942b-e7b1c681247b.PNG)

Pipeline의 lifecycle은 위와 같습니다. 

**Data-centric**한 방법으로 정확도를 향상하는것이 목표이기 때문에 Model을 save하여 버전을 관리하거나 Pushing(Serving)하는 대신, **Evaluator까지만** 도입하여 결과를 분석합니다.



TFX pipeline의 각 구성요소에 대한 자세한 사항은 [여기](https://www.tensorflow.org/tfx/guide?hl=ko) 를 참조해주세요.



![diagram](https://user-images.githubusercontent.com/47979730/123546444-7bae8300-d797-11eb-99e1-fd2e6ccc9e11.PNG)

Pipeline을 실행하는것을 통해 위의 반복과정을 자동화합니다. 이 경우에는, Model의 조정하는 과정 대신 Data를 생성/조정을 수행합니다.

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



**model.py** : 학습에 사용될 Model의 코드입니다. 

**preprocessing.py** : Pipeline의 transform 과정 중 사용하는 전처리 모듈입니다.



**configs.py** : pipeline의 batch size, #epoch 등의 정보가 들어있습니다.

**pipeline.py** : pipeline을 구성하는 모듈입니다.



**interactive.ipynb** : Pipeline을 구성할 경우, Model은 필수적으로 .tfrecord 형식으로 Data를 입력받아야합니다. 이를 위해 Data directory에 있는 이미지들을 .tfrecord로 바꿔주는 Utility 함수를 제공합니다.

또한 전체적인 Pipeline을 kubeflow dashboard가 아니라 notebook의 interactive context를 통해 실행해볼 수 있습니다.



**template.ipynb** : kubernates서버와 Cloud Storage 등 Pipeline을 구성하는데에 필요한 사전 연결 작업을 수행합니다.