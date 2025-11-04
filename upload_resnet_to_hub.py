"""
MedicalNet ResNet3D 모델을 Hugging Face Hub에 업로드하는 스크립트

사용법:
    # 단일 모델 업로드
    python upload_resnet_to_hub.py --model_variant resnet10 --model_name "your-username/medicalnet-resnet3d-10"
    
    # 모든 모델 자동 업로드
    python upload_resnet_to_hub.py --upload_all --username "your-username"

예시:
    python upload_resnet_to_hub.py --model_variant resnet50 --model_name "myuser/medicalnet-resnet3d-50"
"""

import argparse
import os
import shutil
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from safetensors.torch import save_file

from train.resnet_model.configuration_resnet import (
    ResNet3DConfig,
    ResNet3D10Config,
    ResNet3D50Config,
    ResNet3D101Config,
    ResNet3D152Config,
    ResNet3D200Config,
)
from train.resnet_model.modeling_resnet import (
    ResNet3DModel,
    ResNet3DForImageClassification,
    ResNet3D10ForImageClassification,
    ResNet3D50ForImageClassification,
    ResNet3D101ForImageClassification,
    ResNet3D152ForImageClassification,
    ResNet3D200ForImageClassification,
)


# MedicalNet 모델 정보
MEDICALNET_MODELS = {
    "10": {
        "filename": "resnet_10.pth",
        "local_path": "/workspace/train/resnet_pth/resnet_10.pth",
        "config_class": ResNet3D10Config,
        "model_class": ResNet3D10ForImageClassification,
        "depths": [1, 1, 1, 1],
        "layer_type": "basic",
        "description": "MedicalNet ResNet3D-10 pretrained on medical dataset",
    },
    "10-23datasets": {
        "filename": "resnet_10_23dataset.pth",
        "local_path": "/workspace/train/resnet_pth/resnet_10_23dataset.pth",
        "config_class": ResNet3D10Config,
        "model_class": ResNet3D10ForImageClassification,
        "depths": [1, 1, 1, 1],
        "layer_type": "basic",
        "description": "MedicalNet ResNet3D-10 pretrained on 23 medical datasets",
    },
    "resnet50": {
        "filename": "resnet_50.pth",
        "local_path": "/workspace/train/resnet_pth/resnet_50.pth",
        "config_class": ResNet3D50Config,
        "model_class": ResNet3D50ForImageClassification,
        "depths": [3, 4, 6, 3],
        "layer_type": "bottleneck",
        "description": "MedicalNet ResNet3D-50 pretrained on medical dataset",
    },
    "50-23datasets": {
        "filename": "resnet_50_23dataset.pth",
        "local_path": "/workspace/train/resnet_pth/resnet_50_23dataset.pth",
        "config_class": ResNet3D50Config,
        "model_class": ResNet3D50ForImageClassification,
        "depths": [3, 4, 6, 3],
        "layer_type": "bottleneck",
        "description": "MedicalNet ResNet3D-50 pretrained on 23 medical datasets",
    },
    "101": {
        "filename": "resnet_101.pth",
        "local_path": "/workspace/train/resnet_pth/resnet_101.pth",
        "config_class": ResNet3D101Config,
        "model_class": ResNet3D101ForImageClassification,
        "depths": [3, 4, 23, 3],
        "layer_type": "bottleneck",
        "description": "MedicalNet ResNet3D-101 pretrained model",
    },
    "152": {
        "filename": "resnet_152.pth",
        "local_path": "/workspace/train/resnet_pth/resnet_152.pth",
        "config_class": ResNet3D152Config,
        "model_class": ResNet3D152ForImageClassification,
        "depths": [3, 8, 36, 3],
        "layer_type": "bottleneck",
        "description": "MedicalNet ResNet3D-152 pretrained model",
    },
    "200": {
        "filename": "resnet_200.pth",
        "local_path": "/workspace/train/resnet_pth/resnet_200.pth",
        "config_class": ResNet3D200Config,
        "model_class": ResNet3D200ForImageClassification,
        "depths": [3, 24, 36, 3],
        "layer_type": "bottleneck",
        "description": "MedicalNet ResNet3D-200 pretrained model",
    },
}


def get_model_path(local_path: str) -> str:
    """로컬 모델 파일 경로를 확인합니다."""
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {local_path}")
    
    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"  ✅ 모델 파일 확인됨: {os.path.basename(local_path)} ({file_size_mb:.1f} MB)")
    return local_path


def convert_old_keys_to_new(old_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    기존 MedicalNet 모델의 키를 새로운 HuggingFace 스타일 키로 변환합니다.
    
    기존 구조:
    - conv1, bn1 -> resnet3d.embedder.embedder.convolution, normalization
    - maxpool -> resnet3d.embedder.pooler
    - layer1, layer2, layer3, layer4 -> resnet3d.encoder.stages[0-3]
    - avgpool -> resnet3d.pooler
    - fc -> classifier.1
    """
    new_state_dict = {}
    
    for old_key, value in old_state_dict.items():
        new_key = old_key
        
        # conv1 -> embedder.embedder.convolution
        if old_key == "conv1.weight":
            new_key = "resnet3d.embedder.embedder.convolution.weight"
        elif old_key == "conv1.bias":
            new_key = "resnet3d.embedder.embedder.convolution.bias"
        
        # bn1 -> embedder.embedder.normalization
        elif old_key.startswith("bn1."):
            param_name = old_key.replace("bn1.", "")
            new_key = f"resnet3d.embedder.embedder.normalization.{param_name}"
        
        # layer1-4 -> encoder.stages[0-3]
        elif old_key.startswith("layer"):
            # layer1 -> stage 0, layer2 -> stage 1, etc.
            parts = old_key.split(".")
            layer_num = int(parts[0].replace("layer", ""))
            stage_idx = layer_num - 1
            
            # layer1.0.conv1 -> encoder.stages[0].layers.0.layer.0.convolution
            block_idx = parts[1]
            rest = ".".join(parts[2:])
            
            # BasicBlock: conv1, bn1, conv2, bn2, downsample
            # Bottleneck: conv1, bn1, conv2, bn2, conv3, bn3, downsample
            
            if rest.startswith("downsample."):
                # downsample.0 -> shortcut.convolution
                # downsample.1 -> shortcut.normalization
                if "0.weight" in rest or "0.bias" in rest:
                    param = rest.split(".")[-1]
                    new_key = f"resnet3d.encoder.stages.{stage_idx}.layers.{block_idx}.shortcut.convolution.{param}"
                else:
                    param_name = rest.replace("downsample.1.", "")
                    new_key = f"resnet3d.encoder.stages.{stage_idx}.layers.{block_idx}.shortcut.normalization.{param_name}"
            
            elif rest.startswith("conv1"):
                # conv1 -> layer.0.convolution (for BasicBlock) or layer.0.convolution (for Bottleneck)
                param = rest.replace("conv1.", "")
                new_key = f"resnet3d.encoder.stages.{stage_idx}.layers.{block_idx}.layer.0.convolution.{param}"
            
            elif rest.startswith("bn1"):
                # bn1 -> layer.0.normalization
                param = rest.replace("bn1.", "")
                new_key = f"resnet3d.encoder.stages.{stage_idx}.layers.{block_idx}.layer.0.normalization.{param}"
            
            elif rest.startswith("conv2"):
                # conv2 -> layer.1.convolution
                param = rest.replace("conv2.", "")
                new_key = f"resnet3d.encoder.stages.{stage_idx}.layers.{block_idx}.layer.1.convolution.{param}"
            
            elif rest.startswith("bn2"):
                # bn2 -> layer.1.normalization
                param = rest.replace("bn2.", "")
                new_key = f"resnet3d.encoder.stages.{stage_idx}.layers.{block_idx}.layer.1.normalization.{param}"
            
            elif rest.startswith("conv3"):
                # conv3 -> layer.2.convolution (only for Bottleneck)
                param = rest.replace("conv3.", "")
                new_key = f"resnet3d.encoder.stages.{stage_idx}.layers.{block_idx}.layer.2.convolution.{param}"
            
            elif rest.startswith("bn3"):
                # bn3 -> layer.2.normalization (only for Bottleneck)
                param = rest.replace("bn3.", "")
                new_key = f"resnet3d.encoder.stages.{stage_idx}.layers.{block_idx}.layer.2.normalization.{param}"
        
        # fc -> classifier.1
        elif old_key.startswith("fc."):
            param = old_key.replace("fc.", "")
            new_key = f"classifier.1.{param}"
        
        new_state_dict[new_key] = value
    
    return new_state_dict


_MODELS_REGISTERED = False

def register_resnet3d_models():
    """ResNet3D 모델을 AutoClass에 등록"""
    global _MODELS_REGISTERED
    
    if _MODELS_REGISTERED:
        return
    
    # AutoConfig에 등록
    AutoConfig.register("resnet3d", ResNet3DConfig)
    
    # AutoModel에 등록
    AutoModel.register(ResNet3DConfig, ResNet3DModel)
    
    # AutoModelForImageClassification에 등록
    from transformers import AutoModelForImageClassification
    AutoModelForImageClassification.register(ResNet3DConfig, ResNet3DForImageClassification)
    
    _MODELS_REGISTERED = True
    print("✅ ResNet3D 모델이 AutoClass에 등록되었습니다.")


def load_pretrained_weights(model, pth_file: str):
    """사전 학습된 가중치를 모델에 로드하고 safetensors로 변환합니다."""
    device = torch.device("cpu")  # CPU에서 로드하여 메모리 절약
    
    print(f"  📥 PTH 파일 로드 중...")
    pretrained_state_dict = torch.load(pth_file, map_location=device)
    
    # state_dict 키 정리
    if "state_dict" in pretrained_state_dict:
        pretrained_state_dict = pretrained_state_dict["state_dict"]
    
    # DataParallel wrapper 제거
    pretrained_state_dict = {k.replace("module.", ""): v for k, v in pretrained_state_dict.items()}
    
    print(f"  🔄 키 변환 중 (기존 MedicalNet -> HuggingFace 스타일)...")
    # 키 변환
    converted_state_dict = convert_old_keys_to_new(pretrained_state_dict)
    
    # 현재 모델의 state_dict 가져오기
    model_state_dict = model.state_dict()
    
    # 매칭되는 키만 로드
    matched_keys = []
    mismatched_keys = []
    missing_keys = []
    
    for key in converted_state_dict.keys():
        if key in model_state_dict:
            if converted_state_dict[key].shape == model_state_dict[key].shape:
                matched_keys.append(key)
            else:
                mismatched_keys.append(key)
                print(f"     ⚠️  Shape 불일치: {key}")
                print(f"        - 사전학습: {converted_state_dict[key].shape}")
                print(f"        - 현재모델: {model_state_dict[key].shape}")
    
    # 모델에만 있는 새 키 (분류 헤드 등)
    for key in model_state_dict.keys():
        if key not in converted_state_dict:
            missing_keys.append(key)
    
    # 매칭되는 가중치만 로드
    filtered_state_dict = {k: v for k, v in converted_state_dict.items() if k in matched_keys}
    model.load_state_dict(filtered_state_dict, strict=False)
    
    print(f"  ✅ 가중치 로드 완료:")
    print(f"     - 로드된 레이어: {len(matched_keys)}개")
    print(f"     - 새로 초기화된 레이어: {len(missing_keys)}개")
    if mismatched_keys:
        print(f"     - Shape 불일치로 제외: {len(mismatched_keys)}개")
    
    if len(matched_keys) < 10:
        print(f"\n  ⚠️  경고: 로드된 레이어가 매우 적습니다. 키 매핑을 확인하세요.")
        print(f"  샘플 기존 키: {list(pretrained_state_dict.keys())[:3]}")
        print(f"  샘플 변환 키: {list(converted_state_dict.keys())[:3]}")
        print(f"  샘플 모델 키: {list(model_state_dict.keys())[:3]}")
    
    return model


def upload_model_to_hub(
    model_variant: str,
    model_name: str,
    spatial_dims: int = 3,
    num_channels: int = 1,
    num_labels: int = 400,  # MedicalNet의 기본 클래스 수
):
    """
    MedicalNet ResNet3D 모델을 Hugging Face Hub에 업로드
    
    Args:
        model_variant: 모델 변형 (예: 'resnet10', 'resnet50_23datasets')
        model_name: Hub에 업로드할 모델 이름 (예: "username/medicalnet-resnet3d-10")
        spatial_dims: 공간 차원 (3D 의료 영상이므로 3)
        num_channels: 입력 채널 수
        num_labels: 출력 클래스 수
    """
    print("=" * 80)
    print(f"MedicalNet {model_variant.upper()} 모델을 Hugging Face Hub에 업로드 중...")
    print("=" * 80)
    
    if model_variant not in MEDICALNET_MODELS:
        raise ValueError(f"지원하지 않는 모델 변형: {model_variant}")
    
    model_info = MEDICALNET_MODELS[model_variant]
    
    # 1. 로컬 모델 파일 확인
    print(f"\n📂 로컬 모델 파일 확인 중...")
    pth_file = get_model_path(model_info["local_path"])
    
    # 2. Configuration 생성
    print(f"\n📋 Configuration 생성 중...")
    config_class = model_info["config_class"]
    config = config_class(
        spatial_dims=spatial_dims,
        num_channels=num_channels,
        num_labels=num_labels,
    )
    
    print(f"  - Model: ResNet3D-{model_variant}")
    print(f"  - Spatial Dimensions: {config.spatial_dims}D")
    print(f"  - Input Channels: {config.num_channels}")
    print(f"  - Output Classes: {config.num_labels}")
    print(f"  - Depths: {config.depths}")
    print(f"  - Layer Type: {config.layer_type}")
    
    # 3. 모델 생성
    print(f"\n🏗️  모델 생성 중...")
    model_class = model_info["model_class"]
    model = model_class(config)
    
    # 4. 사전 학습된 가중치 로드
    print(f"\n⚙️  사전 학습된 가중치 로드 중...")
    model = load_pretrained_weights(model, pth_file)
    
    # 모델 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 모델 통계:")
    print(f"  - Total Parameters: {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,}")
    
    # 5. 임시 디렉토리에 모델 저장 및 코드 파일 복사
    print(f"\n💾 로컬에 모델 저장 중...")
    temp_dir = f"./temp_{model_variant}"
    
    # 임시 디렉토리가 있으면 삭제
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # 모델과 설정 저장
    model.save_pretrained(temp_dir, safe_serialization=True)
    config.save_pretrained(temp_dir)
    print(f"  ✅ 모델 및 설정 저장 완료: {temp_dir}")
    
    # 6. 모델 코드 파일 복사 (trust_remote_code를 위해 필수)
    print(f"\n📋 모델 코드 파일 복사 중...")
    source_config_file = "train/resnet_model/configuration_resnet.py"
    source_modeling_file = "train/resnet_model/modeling_resnet.py"
    
    shutil.copy2(source_config_file, os.path.join(temp_dir, "configuration_resnet.py"))
    shutil.copy2(source_modeling_file, os.path.join(temp_dir, "modeling_resnet.py"))
    print(f"  ✅ configuration_resnet.py 복사 완료")
    print(f"  ✅ modeling_resnet.py 복사 완료")
    
    # 7. Hub에 업로드
    print(f"\n☁️  Hugging Face Hub에 업로드 중...")
    print(f"  - Model Name: {model_name}")
    print(f"  - Description: {model_info['description']}")
    print(f"  - Format: safetensors")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # 레포지토리 생성 (이미 있으면 무시)
        print(f"\n  🔧 레포지토리 확인/생성 중...")
        try:
            api.create_repo(
                repo_id=model_name,
                repo_type="model",
                exist_ok=True,  # 이미 있으면 무시
                private=False
            )
            print(f"  ✅ 레포지토리 준비 완료")
        except Exception as e:
            print(f"  ⚠️  레포지토리 생성 경고: {e}")
            print(f"  ℹ️  기존 레포지토리에 업로드 시도...")
        
        print(f"\n  📤 전체 폴더 업로드 중...")
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=model_name,
            repo_type="model",
            commit_message=f"Upload {model_variant} model with trust_remote_code support"
        )
        print(f"  ✅ 업로드 완료")
        
        # 8. 임시 디렉토리 삭제
        print(f"\n🗑️  임시 파일 정리 중...")
        shutil.rmtree(temp_dir)
        print(f"  ✅ 정리 완료")
        
        print(f"\n" + "=" * 80)
        print(f"🎉 업로드 성공!")
        print("=" * 80)
        print(f"\n모델 사용 방법:")
        print(f"```python")
        print(f"from transformers import AutoConfig, AutoModelForImageClassification")
        print(f"")
        print(f"config = AutoConfig.from_pretrained('{model_name}', trust_remote_code=True)")
        print(f"model = AutoModelForImageClassification.from_pretrained(")
        print(f"    '{model_name}',")
        print(f"    trust_remote_code=True")
        print(f")")
        print(f"```")
        print(f"\nHub URL: https://huggingface.co/{model_name}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 업로드 실패: {e}")
        print(f"\n💡 다음을 확인하세요:")
        print(f"  1. Hugging Face에 로그인되어 있는지 확인")
        print(f"     터미널에서 실행: huggingface-cli login")
        print(f"  2. 모델 이름이 올바른 형식인지 확인 (username/model-name)")
        print(f"  3. 네트워크 연결 상태 확인")
        
        # 에러 발생 시에도 임시 디렉토리 정리
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        raise


def upload_all_models(username: str, num_labels: int = 400):
    """모든 MedicalNet 모델을 Hub에 업로드"""
    print("\n" + "=" * 80)
    print("모든 MedicalNet ResNet3D 모델을 업로드합니다")
    print("=" * 80)
    
    results = {}
    
    for variant_name in MEDICALNET_MODELS.keys():
        model_name = f"{username}/medicalnet-resnet3d{variant_name.replace('_', '-')}"
        print(f"\n\n{'='*80}")
        print(f"[{list(MEDICALNET_MODELS.keys()).index(variant_name) + 1}/{len(MEDICALNET_MODELS)}] {variant_name} 업로드 시작")
        print(f"{'='*80}")
        
        try:
            success = upload_model_to_hub(
                model_variant=variant_name,
                model_name=model_name,
                num_labels=num_labels,
            )
            results[variant_name] = "✅ 성공"
        except Exception as e:
            print(f"❌ {variant_name} 업로드 실패: {e}")
            results[variant_name] = f"❌ 실패: {str(e)[:50]}"
            continue
    
    # 최종 결과 출력
    print("\n\n" + "=" * 80)
    print("업로드 결과 요약")
    print("=" * 80)
    for variant, status in results.items():
        print(f"  {variant:25s} : {status}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="MedicalNet ResNet3D 모델을 Hugging Face Hub에 업로드"
    )
    
    # 단일 모델 업로드 옵션
    parser.add_argument(
        "--model_variant",
        type=str,
        choices=list(MEDICALNET_MODELS.keys()),
        help="업로드할 모델 변형 (예: 'resnet10', 'resnet50_23datasets')",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Hub에 업로드할 모델 이름 (예: 'username/medicalnet-resnet3d-10')",
    )
    
    # 전체 모델 업로드 옵션
    parser.add_argument(
        "--upload_all",
        action="store_true",
        help="모든 MedicalNet 모델을 자동으로 업로드",
    )
    parser.add_argument(
        "--username",
        type=str,
        help="Hugging Face 사용자명 (--upload_all 사용 시 필수)",
    )
    
    # 공통 옵션
    parser.add_argument(
        "--spatial_dims",
        type=int,
        default=3,
        help="공간 차원 (기본값: 3)",
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=1,
        help="입력 채널 수 (기본값: 1)",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=400,
        help="출력 클래스 수 (기본값: 400, MedicalNet 사전학습)",
    )
    
    args = parser.parse_args()
    
    # 사용 가능한 모델 목록 출력
    print("\n사용 가능한 MedicalNet 모델:")
    for variant, info in MEDICALNET_MODELS.items():
        print(f"  - {variant:25s} : {info['description']}")
    print()
    
    if args.upload_all:
        if not args.username:
            parser.error("--upload_all을 사용할 때는 --username이 필수입니다")
        upload_all_models(args.username, args.num_labels)
    elif args.model_variant and args.model_name:
        upload_model_to_hub(
            model_variant=args.model_variant,
            model_name=args.model_name,
            spatial_dims=args.spatial_dims,
            num_channels=args.num_channels,
            num_labels=args.num_labels,
        )
    else:
        parser.error("--model_variant와 --model_name을 함께 지정하거나, --upload_all과 --username을 지정해야 합니다")


if __name__ == "__main__":
    main()
