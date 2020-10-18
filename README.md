# Student-Teacher anomaly detection
This is an implementation of the paper [Uninformed Students: Student–Teacher Anomaly Detection
with Discriminative Latent Embeddings](https://arxiv.org/pdf/1911.02357v2.pdf). 

## How to use

* Download a dataset from [MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad/) and extract it under the **/data** folder. Let us download the **carpet** dataset for the example. You might need to run
```
chmod -R u+rw data
```

* Run the mvtec_dataset.sh script to prepare the dataset in the correct format. 
```
./mvtec_dataset.sh carpet
```


* (Optional) Run resnet18_training.py script to train resnet18 further on your dataset
```
cd src
python3 resnet18_training.py carpet
```

* Run teacher_training.py to distil the knowledge of resnet18 on a smaller neural network. This will speed up the processing of images. This neural network, called the Teacher, outputs a 512-dimensional description vector for each patch of size **65x65** of the image
```
cd src
python3 teacher_training.py carpet
```

* Run students_training.py to train a set of M=3 students against the teacher network. The training of the students is done on an anomaly-free dataset. We expect them to generalize poorly in images containing anomalies
```
cd src
python3 students_training.py carpet
```

* Run anomaly_detection.py to obtain an anomaly map for each image of the test set. An anomaly map is computed using the variance of Students predictions and the error between Students predictions and Teacher.
```
cd src
python3 anomaly_detection.py carpet
```

## Results
![alt text](https://github.com/denguir/student-teacher-anomaly-detection/blob/master/results/anomaly_carpet_res1.png)
![alt text](https://github.com/denguir/student-teacher-anomaly-detection/blob/master/results/anomaly_carpet_res2.png)

And more results are available under **/result** folder

## Expected folder structure
├── data   
│   ├── carpet  
│   └── hazelnut  
├── docs  
│   ├── 9245_FastCNNFeature_BMVC.pdf  
│   ├── anomaly_detection.pdf  
│   └── anomaly_detection_summary.pdf  
├── model   
│   ├── carpet  
│   └── hazelnut  
├── mvtec_dataset.py  
├── mvtec_dataset.sh  
├── README.md  
├── results  
│   ├── anomaly_carpet_res1.png  
│   ├── anomaly_carpet_res2.png  
│   ├── anomaly_carpet_res3.png  
│   ├── anomaly_carpet_res4.png  
│   ├── anomaly_hazelnul_res2.png  
│   └── anomaly_hazelnut_res1.png  
└── src  
    ├── AnomalyDataset.py  
    ├── anomaly_detection.py  
    ├── AnomalyNet.py  
    ├── AnomalyResnet18.py  
    ├── ExtendedAnomalyNet.py  
    ├── FDFEAnomalyNet.py  
    ├── FDFE.py  
    ├── resnet18_training.py  
    ├── students_training.py  
    ├── teacher_training.py  
    └── utils.py  


## References

### Original paper
* [https://arxiv.org/pdf/1911.02357v2.pdf](https://arxiv.org/pdf/1911.02357v2.pdf)

### Dataset 
* [https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/mvtec_ad.pdf](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/mvtec_ad.pdf)

### Fast Dense Feature Extraction
* [https://www.dfki.de/fileadmin/user_upload/import/9245_FastCNNFeature_BMVC.pdf](https://www.dfki.de/fileadmin/user_upload/import/9245_FastCNNFeature_BMVC.pdf)
* [https://github.com/erezposner/Fast_Dense_Feature_Extraction](https://github.com/erezposner/Fast_Dense_Feature_Extraction)