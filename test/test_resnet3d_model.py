"""
ResNet3D model structure and weight loading test script
"""

import torch
from resnet_model.configuration_resnet import ResNet3D10Config, ResNet3D50Config
from resnet_model.modeling_resnet import (
    ResNet3D10ForImageClassification,
    ResNet3D50ForImageClassification,
)


def test_model_structure():
    """Test model structure"""
    print("=" * 80)
    print("ResNet3D Model Structure Test")
    print("=" * 80)
    
    # ResNet3D-10 test
    print("\n1. ResNet3D-10 Model Creation Test")
    config10 = ResNet3D10Config(
        spatial_dims=3,
        num_channels=1,
        num_labels=2,
    )
    model10 = ResNet3D10ForImageClassification(config10)
    
    total_params = sum(p.numel() for p in model10.parameters())
    print(f"   ✅ Model created successfully")
    print(f"   - Total Parameters: {total_params:,}")
    print(f"   - Config: {config10.layer_type}, depths={config10.depths}")
    
    # ResNet3D-50 test
    print("\n2. ResNet3D-50 Model Creation Test")
    config50 = ResNet3D50Config(
        spatial_dims=3,
        num_channels=1,
        num_labels=400,
    )
    model50 = ResNet3D50ForImageClassification(config50)
    
    total_params = sum(p.numel() for p in model50.parameters())
    print(f"   ✅ Model created successfully")
    print(f"   - Total Parameters: {total_params:,}")
    print(f"   - Config: {config50.layer_type}, depths={config50.depths}")
    
    return model10, model50


def test_forward_pass(model, model_name="ResNet3D"):
    """Test forward pass"""
    print(f"\n3. {model_name} Forward Pass Test")
    
    # Create dummy input (batch=2, channels=1, depth=16, height=64, width=64)
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 16, 64, 64)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
    
    print(f"   ✅ Forward pass successful")
    print(f"   - Input shape: {input_tensor.shape}")
    print(f"   - Logits shape: {outputs.logits.shape}")
    print(f"   - Expected: ({batch_size}, 400)")
    
    assert outputs.logits.shape == (batch_size, 400), "Output shape differs from expected!"
    print(f"   ✅ Output shape validation completed")


def test_state_dict_keys(model, model_name="ResNet3D"):
    """Check state dict key structure"""
    print(f"\n4. {model_name} State Dict Key Structure Check")
    
    state_dict = model.state_dict()
    
    # Check main keys
    embedder_keys = [k for k in state_dict.keys() if "embedder" in k]
    encoder_keys = [k for k in state_dict.keys() if "encoder" in k]
    classifier_keys = [k for k in state_dict.keys() if "classifier" in k]
    
    print(f"   - Embedder related keys: {len(embedder_keys)}")
    print(f"     Sample: {embedder_keys[:2]}")
    print(f"   - Encoder related keys: {len(encoder_keys)}")
    print(f"     Sample: {encoder_keys[:2]}")
    print(f"   - Classifier related keys: {len(classifier_keys)}")
    print(f"     Sample: {classifier_keys}")
    
    print(f"\n   Total parameter keys: {len(state_dict)}")


def test_old_weight_loading():
    """Test loading old PTH file (if file exists)"""
    import os
    
    print("\n5. Old PTH File Loading Test")
    
    pth_file = "resnet_pth/resnet_10.pth"
    
    if not os.path.exists(pth_file):
        print(f"   ⚠️  PTH file not found: {pth_file}")
        print(f"   Skipping test.")
        return
    
    print(f"   📥 Loading PTH file...")
    checkpoint = torch.load(pth_file, map_location="cpu")
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Remove DataParallel wrapper
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    print(f"   ✅ PTH file loaded successfully")
    print(f"   - Total keys: {len(state_dict)}")
    print(f"   - Sample keys:")
    for i, key in enumerate(list(state_dict.keys())[:5]):
        print(f"     {i+1}. {key}: {state_dict[key].shape}")
    
    # Key structure analysis
    conv_keys = [k for k in state_dict.keys() if "conv" in k]
    bn_keys = [k for k in state_dict.keys() if "bn" in k]
    layer_keys = [k for k in state_dict.keys() if "layer" in k]
    fc_keys = [k for k in state_dict.keys() if "fc" in k]
    
    print(f"\n   Key structure analysis:")
    print(f"   - conv related: {len(conv_keys)}")
    print(f"   - bn related: {len(bn_keys)}")
    print(f"   - layer related: {len(layer_keys)}")
    print(f"   - fc related: {len(fc_keys)}")


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "ResNet3D Model Test Starting" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        # 1. Test model structure
        model10, model50 = test_model_structure()
        
        # 2. Test forward pass
        test_forward_pass(model10, "ResNet3D-10")
        test_forward_pass(model50, "ResNet3D-50")
        
        # 3. Check state dict key structure
        test_state_dict_keys(model10, "ResNet3D-10")
        
        # 4. Test loading old PTH file
        test_old_weight_loading()
        
        print("\n" + "=" * 80)
        print("🎉 All tests passed!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Upload to Hub using upload_resnet_to_hub.py")
        print("  2. Example:")
        print("     python upload_resnet_to_hub.py \\")
        print("       --model_variant resnet10 \\")
        print("       --model_name 'your-username/medicalnet-resnet3d-10'")
        print()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

