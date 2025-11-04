"""
MedicalNet model download and test script

Usage:
    python test_medicalnet_download.py --model_variant resnet10
"""

import argparse
import errno
import os
from typing import Optional

import gdown
import torch

from train.resnet_model.configuration_resnet import (
    ResNet10Config,
    ResNet50Config,
    ResNet101Config,
    ResNet152Config,
    ResNet200Config,
)
from train.resnet_model.modeling_resnet import (
    ResNet10ForImageClassification,
    ResNet50ForImageClassification,
    ResNet101ForImageClassification,
    ResNet152ForImageClassification,
    ResNet200ForImageClassification,
)


MEDICALNET_MODELS = {
    "resnet10": {
        "url": "https://drive.google.com/uc?export=download&id=1lCEK_K5q90YaOtyfkGAjUCMrqcQZUYV0",
        "filename": "resnet_10.pth",
        "config_class": ResNet10Config,
        "model_class": ResNet10ForImageClassification,
    },
    "resnet10_23datasets": {
        "url": "https://drive.google.com/uc?export=download&id=1HLpyQ12SmzmCIFjMcNs4j3Ijyy79JYLk",
        "filename": "resnet_10_23dataset.pth",
        "config_class": ResNet10Config,
        "model_class": ResNet10ForImageClassification,
    },
    "resnet50": {
        "url": "https://drive.google.com/uc?export=download&id=1E7005_ZT_z6tuPpPNRvYkMBWzAJNMIIC",
        "filename": "resnet_50.pth",
        "config_class": ResNet50Config,
        "model_class": ResNet50ForImageClassification,
    },
    "resnet50_23datasets": {
        "url": "https://drive.google.com/uc?export=download&id=1qXyw9S5f-6N1gKECDfMroRnPZfARbqOP",
        "filename": "resnet_50_23dataset.pth",
        "config_class": ResNet50Config,
        "model_class": ResNet50ForImageClassification,
    },
    "resnet101": {
        "url": "https://drive.google.com/uc?export=download&id=1mMNQvhlaS-jmnbyqdniGNSD5aONIidKt",
        "filename": "resnet_101.pth",
        "config_class": ResNet101Config,
        "model_class": ResNet101ForImageClassification,
    },
    "resnet152": {
        "url": "https://drive.google.com/uc?export=download&id=1Lixxc9YsZZqAl3mnAh7PwT8c3sTXoinE",
        "filename": "resnet_152.pth",
        "config_class": ResNet152Config,
        "model_class": ResNet152ForImageClassification,
    },
    "resnet200": {
        "url": "https://drive.google.com/uc?export=download&id=13BGtYw2fkvDSlx41gOZ5qTFhhrDB_zXr",
        "filename": "resnet_200.pth",
        "config_class": ResNet200Config,
        "model_class": ResNet200ForImageClassification,
    },
}


def download_model(url: str, filename: str, model_dir: Optional[str] = None) -> str:
    """Download MedicalNet model."""
    if model_dir is None:
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, "medicalnet")

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        print(f"📥 Downloading: {filename}")
        gdown.download(url=url, output=cached_file, quiet=False)
        print(f"✅ Download completed: {cached_file}")
    else:
        print(f"✅ Already downloaded: {cached_file}")
    
    return cached_file


def test_model(model_variant: str):
    """Download and test the model."""
    print("=" * 80)
    print(f"MedicalNet {model_variant.upper()} Model Test")
    print("=" * 80)
    
    if model_variant not in MEDICALNET_MODELS:
        print(f"❌ Unsupported model: {model_variant}")
        print(f"Available models: {list(MEDICALNET_MODELS.keys())}")
        return
    
    model_info = MEDICALNET_MODELS[model_variant]
    
    # 1. Download model
    print("\n1️⃣ Downloading pretrained weights")
    cached_file = download_model(
        url=model_info["url"],
        filename=model_info["filename"],
    )
    
    # 2. Create configuration
    print("\n2️⃣ Creating configuration")
    config_class = model_info["config_class"]
    config = config_class(
        spatial_dims=3,
        in_channels=1,
        out_channels=400,  # MedicalNet default
    )
    print(f"  - Layers: {config.layers}")
    print(f"  - Block Type: {config.block_type}")
    
    # 3. Create model
    print("\n3️⃣ Creating model")
    model_class = model_info["model_class"]
    model = model_class(config)
    
    # 4. Load weights
    print("\n4️⃣ Loading pretrained weights")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Device: {device}")
    
    pretrained_state_dict = torch.load(cached_file, map_location=device)
    
    # Clean state_dict keys
    if "state_dict" in pretrained_state_dict:
        pretrained_state_dict = pretrained_state_dict["state_dict"]
    
    # Remove DataParallel wrapper
    pretrained_state_dict = {k.replace("module.", ""): v for k, v in pretrained_state_dict.items()}
    
    # Current model's state_dict
    model_state_dict = model.state_dict()
    
    # Load only matching keys
    matched_keys = []
    for key in pretrained_state_dict.keys():
        if key in model_state_dict:
            if pretrained_state_dict[key].shape == model_state_dict[key].shape:
                matched_keys.append(key)
    
    filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in matched_keys}
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    print(f"  ✅ Loaded layers: {len(matched_keys)}")
    print(f"  ⚠️  Missing keys: {len(missing_keys)}")
    if missing_keys[:5]:
        print(f"     Examples: {missing_keys[:5]}")
    print(f"  ⚠️  Unexpected keys: {len(unexpected_keys)}")
    
    # 5. Model parameter statistics
    print("\n5️⃣ Model statistics")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total Parameters: {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,}")
    print(f"  - Model Size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # 6. Inference test
    print("\n6️⃣ Testing inference")
    model.eval()
    with torch.no_grad():
        # Create dummy input (batch_size=1, channels=1, D=32, H=32, W=32)
        dummy_input = torch.randn(1, 1, 32, 32, 32).to(device)
        model = model.to(device)
        
        print(f"  - Input shape: {dummy_input.shape}")
        
        try:
            outputs = model(pixel_values=dummy_input)
            logits = outputs['logits']
            print(f"  ✅ Output shape: {logits.shape}")
            print(f"  ✅ Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
            
            # Predicted class
            predicted_class = logits.argmax(dim=-1).item()
            print(f"  ✅ Predicted class: {predicted_class}")
            
        except Exception as e:
            print(f"  ❌ Inference failed: {e}")
            raise
    
    print("\n" + "=" * 80)
    print("✅ Test completed!")
    print("=" * 80)


def test_all_models():
    """Test all models sequentially."""
    print("\n" + "=" * 80)
    print("Testing All MedicalNet Models")
    print("=" * 80)
    
    results = {}
    
    for variant in MEDICALNET_MODELS.keys():
        print(f"\n\n{'='*80}")
        print(f"Testing: {variant}")
        print(f"{'='*80}")
        
        try:
            test_model(variant)
            results[variant] = "✅ Success"
        except Exception as e:
            print(f"❌ {variant} test failed: {e}")
            results[variant] = f"❌ Failed"
            continue
    
    # Result summary
    print("\n\n" + "=" * 80)
    print("Test Results Summary")
    print("=" * 80)
    for variant, status in results.items():
        print(f"  {variant:25s} : {status}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Download and test MedicalNet ResNet models"
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        choices=list(MEDICALNET_MODELS.keys()),
        help="Model variant to test",
    )
    parser.add_argument(
        "--test_all",
        action="store_true",
        help="Test all models",
    )
    
    args = parser.parse_args()
    
    if args.test_all:
        test_all_models()
    elif args.model_variant:
        test_model(args.model_variant)
    else:
        parser.error("Must specify --model_variant or --test_all")


if __name__ == "__main__":
    main()

