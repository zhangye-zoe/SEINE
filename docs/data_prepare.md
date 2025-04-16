# Dataset Prepare

It is recommended to symlink the dataset root to `$ROOT/data`. If your folder structure is different, you may need to change the corresponding paths in config files.

```None
data
└── consep
    ├── train
    │   ├── xxx.tif
    │   ├── xxx.mat
    │   └── xxx.png
    ├── valid
    │   ├── xxx.png
    │   ├── xxx.mat
    │   └── xxx.png
    ├── test
    │   ├── xxx.png
    │   ├── xxx.mat
    │   └── xxx.png
    ├── train.txt
    ├── valid.txt
    └── test.txt


```


## CoNSeP Nuclei Segmentation Dataset

***!!Attention*** part of nuclei tissue images may have 4 channels (R, G, B, Alpha)

1. Download CoNSeP dataset from [homepage](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/);
2. Uncompress them into `data/consep`;
3. Run convertion script: `python tools/convert_dataset/consep.py data/consep -c 300`;


