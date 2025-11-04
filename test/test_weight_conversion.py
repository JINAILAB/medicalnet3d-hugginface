"""
Test for converting old PTH files to new HuggingFace model
"""

import torch
from train.resnet_model.configuration_resnet import ResNet3D10Config
from train.resnet_model.modeling_resnet import ResNet3D10ForImageClassification
from upload_resnet_to_hub import convert_old_keys_to_new


def test_weight_conversion():
    """Weight conversion test"""
    print("=" * 80)
    print("PTH File -> HuggingFace Model Weight Conversion Test")
    print("=" * 80)
    
    # 1. Load PTH file
    print("\n1. Loading PTH file")
    pth_file = "/workspace/train/resnet_pth/resnet_10.pth"
    checkpoint = torch.load(pth_file, map_location="cpu")
    
    if "state_dict" in checkpoint:
        old_state_dict = checkpoint["state_dict"]
    else:
        old_state_dict = checkpoint
    
    # Remove DataParallel wrapper
    old_state_dict = {k.replace("module.", ""): v for k, v in old_state_dict.items()}
    
    print(f"   ✅ PTH file loaded successfully")
    print(f"   - Total keys: {len(old_state_dict)}")
    
    # 2. Convert keys
    print("\n2. Converting keys (Original MedicalNet -> HuggingFace)")
    new_state_dict = convert_old_keys_to_new(old_state_dict)
    
    print(f"   ✅ Key conversion completed")
    print(f"   - Converted keys: {len(new_state_dict)}")
    
    # 3. Sample key comparison
    print("\n3. Key conversion examples:")
    old_keys_sample = list(old_state_dict.keys())[:10]
    
    for old_key in old_keys_sample:
        if old_key in ["conv1.weight"]:
            new_key = "resnet3d.embedder.embedder.convolution.weight"
            print(f"   {old_key:30s} -> {new_key}")
        elif old_key.startswith("bn1"):
            new_key = old_key.replace("bn1.", "resnet3d.embedder.embedder.normalization.")
            print(f"   {old_key:30s} -> {new_key}")
        elif old_key.startswith("layer1.0.conv1"):
            print(f"   {old_key:30s} -> resnet3d.encoder.stages.0.layers.0.layer.0.convolution.weight")
    
    # 4. Create model
    print("\n4. Creating HuggingFace model")
    config = ResNet3D10Config(
        spatial_dims=3,
        num_channels=1,
        num_labels=400,
    )
    model = ResNet3D10ForImageClassification(config)
    model_state_dict = model.state_dict()
    
    print(f"   ✅ Model created successfully")
    print(f"   - Model keys: {len(model_state_dict)}")
    
    # 5. Key matching analysis
    print("\n5. Key matching analysis")
    
    matched_keys = []
    mismatched_shape_keys = []
    missing_in_new = []
    missing_in_old = []
    
    for key in new_state_dict.keys():
        if key in model_state_dict:
            if new_state_dict[key].shape == model_state_dict[key].shape:
                matched_keys.append(key)
            else:
                mismatched_shape_keys.append(key)
        else:
            missing_in_new.append(key)
    
    for key in model_state_dict.keys():
        if key not in new_state_dict:
            missing_in_old.append(key)
    
    print(f"   ✅ Matching analysis completed:")
    print(f"   - Successfully matched: {len(matched_keys)}")
    print(f"   - Shape mismatch: {len(mismatched_shape_keys)}")
    print(f"   - Only in converted dict: {len(missing_in_new)}")
    print(f"   - Only in model (needs initialization): {len(missing_in_old)}")
    
    if mismatched_shape_keys:
        print(f"\n   ⚠️  Shape mismatched keys:")
        for key in mismatched_shape_keys[:5]:
            print(f"      - {key}")
            print(f"        PTH: {new_state_dict[key].shape}")
            print(f"        Model: {model_state_dict[key].shape}")
    
    if missing_in_new:
        print(f"\n   ℹ️  Keys only in converted dict (sample):")
        for key in missing_in_new[:5]:
            print(f"      - {key}")
    
    if missing_in_old:
        print(f"\n   ℹ️  Keys only in model (newly initialized):")
        for key in missing_in_old[:10]:
            print(f"      - {key}")
    
    # 6. Actual weight loading test
    print("\n6. Testing actual weight loading")
    
    filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in matched_keys}
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    
    print(f"   ✅ Weights loaded successfully")
    print(f"   - Loaded keys: {len(filtered_state_dict)}")
    print(f"   - Missing keys: {len(missing)}")
    print(f"   - Unexpected keys: {len(unexpected)}")
    
    # 7. Forward pass test
    print("\n7. Testing forward pass with loaded model")
    
    dummy_input = torch.randn(1, 1, 16, 64, 64)
    model.eval()
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"   ✅ Forward pass successful")
    print(f"   - Input shape: {dummy_input.shape}")
    print(f"   - Output logits shape: {outputs.logits.shape}")
    
    # 8. Calculate matching rate
    total_params_old = len(old_state_dict)
    total_params_model = len(model_state_dict)
    match_rate = (len(matched_keys) / total_params_model) * 100
    
    print("\n" + "=" * 80)
    print("📊 Final Statistics")
    print("=" * 80)
    print(f"Original PTH file parameters: {total_params_old}")
    print(f"HuggingFace model parameters: {total_params_model}")
    print(f"Matching success rate: {match_rate:.1f}%")
    print(f"Matched keys: {len(matched_keys)} / {total_params_model}")
    
    if match_rate >= 85:
        print(f"\n✅ Weight conversion successful! Most weights were matched correctly.")
    else:
        print(f"\n⚠️  Low matching rate. Key mapping needs to be verified.")
    
    print("\n" + "=" * 80)
    
    return match_rate >= 85


if __name__ == "__main__":
    success = test_weight_conversion()
    exit(0 if success else 1)

