"""
macOS 최적화된 디바이스 유틸리티
Apple Silicon (M1/M2/M3) GPU 가속 지원
"""

import torch
import platform
import psutil

def get_optimal_device():
    """
    크로스 플랫폼 최적화된 디바이스 감지
    - 모든 플랫폼: GPU 최우선! (CUDA 또는 MPS) → CPU
    """
    
    # 시스템 정보 확인
    system_info = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'processor': platform.processor()
    }
    
    print(f"🖥️  시스템: {system_info['platform']} {system_info['machine']}")
    
    # 전체 플랫폼에서 GPU 최우선
    
    # 1. CUDA 확인 (Ubuntu/Linux/Windows - NVIDIA GPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA 사용 가능 - GPU: {gpu_name}")
        return device, "CUDA"
    
    # 2. MPS 확인 (macOS - Apple Silicon)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ MPS (Apple Silicon GPU) 사용 가능 - GPU 가속 활성화")
        return device, "MPS"
    
    # 3. CPU fallback (모든 플랫폼 공통 - GPU 없을 때만)
    else:
        device = torch.device('cpu')
        print("⚠️  GPU 사용 불가 - CPU 사용")
        
        # 플랫폼별 CPU 최적화 안내
        if system_info['platform'] == 'Darwin':
            print("💡 macOS CPU 최적화 활성화")
        elif system_info['platform'] == 'Linux':
            print("💡 Linux CPU 최적화 활성화")
            
        return device, "CPU"

def check_gpu_memory(device):
    """GPU 메모리 정보 확인 (크로스 플랫폼)"""
    try:
        if device.type == 'cuda':
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated_memory = torch.cuda.memory_allocated(0) / 1e9
            print(f"🚀 CUDA 메모리: {total_memory:.1f} GB (사용중: {allocated_memory:.1f} GB)")
            
        elif device.type == 'mps':
            # macOS 통합 메모리 정보
            total_memory = psutil.virtual_memory().total / 1e9
            print(f"🍎 MPS 메모리: {total_memory:.1f} GB (통합 메모리)")
            
        else:
            # CPU 메모리 정보 (Linux/Ubuntu/macOS 공통)
            memory_info = psutil.virtual_memory()
            total_memory = memory_info.total / 1e9
            available_memory = memory_info.available / 1e9
            print(f"🖥️  시스템 메모리: {total_memory:.1f} GB (사용가능: {available_memory:.1f} GB)")
            
    except Exception as e:
        print(f"⚠️  메모리 정보 확인 실패: {e}")

def setup_training_device():
    """학습용 디바이스 설정 및 최적화 (크로스 플랫폼)"""
    
    device, device_type = get_optimal_device()
    
    # 디바이스별 최적화 설정
    if device_type == "MPS":
        # MPS 최적화 설정 (macOS)
        try:
            # MPS 메모리 정리 시도 (있는 경우만)
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
            elif hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        except (AttributeError, RuntimeError):
            # MPS 메모리 정리가 지원되지 않는 경우 무시
            pass
        print("🍎 Apple Silicon 최적화 설정 적용")
        
    elif device_type == "CUDA":
        # CUDA 최적화 설정 (Ubuntu/Linux/Windows)
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # 성능 향상을 위해
            print("🚀 CUDA 최적화 설정 적용")
        except Exception as e:
            print(f"⚠️  CUDA 최적화 설정 실패: {e}")
        
    else:
        # CPU 최적화 설정 (모든 플랫폼)
        try:
            # CPU 코어 수에 따른 스레드 수 조정
            import multiprocessing
            num_cores = multiprocessing.cpu_count()
            optimal_threads = min(num_cores, 8)  # 최대 8개로 제한
            torch.set_num_threads(optimal_threads)
            print(f"🖥️  CPU 최적화 설정 적용 (스레드: {optimal_threads}/{num_cores})")
        except Exception as e:
            print(f"⚠️  CPU 최적화 설정 실패: {e}")
    
    # 메모리 정보 출력
    check_gpu_memory(device)
    
    return device, device_type

def get_dataloader_config(device_type):
    """디바이스 타입에 따른 DataLoader 최적화 설정 (크로스 플랫폼)"""
    
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    
    if device_type == "MPS":
        # Apple Silicon 최적화 (macOS)
        return {
            "pin_memory": False,  # MPS는 pin_memory 불필요
            "num_workers": 0,     # MPS는 멀티프로세싱 이슈 방지
        }
    elif device_type == "CUDA":
        # CUDA 최적화 (Ubuntu/Linux/Windows)
        optimal_workers = min(num_cores // 2, 8)  # 코어 수의 절반, 최대 8
        return {
            "pin_memory": True,   # CUDA는 pin_memory 효과적
            "num_workers": optimal_workers,  # 병렬 데이터 로딩
        }
    else:
        # CPU 최적화 (모든 플랫폼)
        optimal_workers = min(num_cores // 4, 4)  # CPU에서는 적당한 수준
        return {
            "pin_memory": False,  # CPU는 pin_memory 불필요
            "num_workers": optimal_workers,
        }

# 테스트 함수
def test_device():
    """디바이스 테스트"""
    print("🔍 디바이스 감지 및 테스트")
    print("=" * 50)
    
    try:
        device, device_type = setup_training_device()
    except Exception as e:
        print(f"⚠️  디바쒰스 설정 오류: {e}")
        print("📋 CPU 모드로 fallback...")
        device = torch.device('cpu')
        device_type = "CPU"
    
    print(f"\n🎯 최종 선택된 디바이스: {device} ({device_type})")
    
    # 간단한 텐서 연산 테스트
    try:
        print("\n🧪 디바이스 테스트 중...")
        test_tensor = torch.randn(1000, 1000).to(device)
        result = torch.matmul(test_tensor, test_tensor)
        print("✅ 텐서 연산 테스트 성공!")
        
        # 메모리 사용량 확인
        if device_type == "CUDA":
            memory_used = torch.cuda.memory_allocated(0) / 1e6
            print(f"📊 GPU 메모리 사용량: {memory_used:.1f} MB")
            
    except Exception as e:
        print(f"❌ 디바이스 테스트 실패: {e}")
        print("💡 CPU로 fallback 권장")
        device = torch.device('cpu')
        device_type = "CPU"
    
    # DataLoader 설정 권장사항
    dataloader_config = get_dataloader_config(device_type)
    print(f"\n⚙️  권장 DataLoader 설정:")
    for key, value in dataloader_config.items():
        print(f"   {key}: {value}")
    
    return device, device_type

if __name__ == "__main__":
    test_device()
