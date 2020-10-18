# Student-Teacher anomaly detection
This is an implementation of the paper [Uninformed Students: Student–Teacher Anomaly Detection
with Discriminative Latent Embeddings](https://arxiv.org/pdf/1911.02357v2.pdf)

## Expected folder structure
├── data
│   ├── carpet
│   └── hazelnut
├── docs
│   ├── 9245_FastCNNFeature_BMVC.pdf
│   └── anomaly_detection.pdf
├── model
│   ├── brain
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
    ├── __pycache__
    ├── resnet18_training.py
    ├── students_training.py
    ├── teacher_training.py
    └── utils.py
