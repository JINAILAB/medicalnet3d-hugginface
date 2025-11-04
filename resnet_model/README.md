---
library_name: transformers
tags:
- MedicalNet
- medical images
- medical
- 3D
- Med3D
license: mit
datasets:
- TencentMedicalNet/MRBrains18
language:
- en
base_model:
- TencentMedicalNet/MedicalNet-Resnet10
thumbnail: "https://github.com/Tencent/MedicalNet/blob/master/images/logo.png?raw=true"

---
# MedicalNet for classifciation

The MedicalNet project aggregated the dataset with diverse modalities, target organs, and pathologies to to build relatively large datasets. Based on this dataset, a series of 3D-ResNet pre-trained models and corresponding transfer-learning training code are provided. 

This repository is an unofficial implementation of Tencent’s Med3D model ([Med3D: Transfer Learning for 3D Medical Image Analysis](https://arxiv.org/abs/1904.00625)), originally developed for segmentation tasks.
It has been adapted for classification tasks using the 3D-ResNet backbone and made compatible with the Hugging Face library.

---

## License
MedicalNet is released under the MIT License (refer to the LICENSE file for detailso).

---

## Citing MedicalNet
If you use this code or pre-trained models, please cite the following:
```
    @article{chen2019med3d,
        title={Med3D: Transfer Learning for 3D Medical Image Analysis},
        author={Chen, Sihong and Ma, Kai and Zheng, Yefeng},
        journal={arXiv preprint arXiv:1904.00625},
        year={2019}
    }
```

---

## Model Sources

- Repository: https://github.com/Tencent/MedicalNet (original)
- Unofficial Torch Hub Wrapper: https://github.com/Warvito/MedicalNet-models

---

## How to Get Started with the Model

```python
from transformers import AutoConfig, AutoModelForImageClassification
import torch
config = AutoConfig.from_pretrained(
    'nwirandx/medicalnet-resnet3d50',
    trust_remote_code=True
)
# use a model from scratch
# model = AutoModelForImageClassification.from_config(
#     config,
#     trust_remote_code=True
# )

# use pretrained model
model = AutoModelForImageClassification.from_pretrained(
    'nwirandx/medicalnet-resnet3d50',
    trust_remote_code=True
)
x = torch.randn(1, 1, 64, 64, 64)  # Example 3D volume
outputs = model(x)
```

---

## MedicalNet Model Family

**Original MedicalNet Series (Tencent on Hugging Face)**

- [TencentMedicalNet/MedicalNet-Resnet10](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10)
- [TencentMedicalNet/MedicalNet-Resnet18](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18)
- [TencentMedicalNet/MedicalNet-Resnet34](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet34)
- [TencentMedicalNet/MedicalNet-Resnet50](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50)
- [TencentMedicalNet/MedicalNet-Resnet101](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet101)
- [TencentMedicalNet/MedicalNet-Resnet152](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet152)
- [TencentMedicalNet/MedicalNet-Resnet200](https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet200)

**Unonfficial Versions of the MedcialNet Classifcation Model Series**

- [nwirandx/medicalnet-resnet3d10](https://huggingface.co/nwirandx/medicalnet-resnet3d10)
- [nwirandx/medicalnet-resnet3d10_23datasets](https://huggingface.co/nwirandx/medicalnet-resnet3d10_23datasets)
- [nwirandx/medicalnet-resnet3d50](https://huggingface.co/nwirandx/medicalnet-resnet3d50)
- [nwirandx/medicalnet-resnet3d50_23datasets](https://huggingface.co/nwirandx/medicalnet-resnet3d50_23datasets)
- [nwirandx/medicalnet-resnet3d101](https://huggingface.co/nwirandx/medicalnet-resnet3d101)
- [nwirandx/medicalnet-resnet3d152](https://huggingface.co/nwirandx/medicalnet-resnet3d152)
- [nwirandx/medicalnet-resnet3d200](https://huggingface.co/nwirandx/medicalnet-resnet3d200)
