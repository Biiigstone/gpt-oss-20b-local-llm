
WSL(Windows Subsystem for Linux) 

### WSL 설치
1. WSL 설치
	PowerShell
	`$ wsl --install`


### GPU 설정
1. Window에 그래픽 드라이버 설치 : GPU 하드웨어를 제어하는 일은 윈도우의 드라이버이다. 따라서 NVIDIA 드라이버가 윈도우에 설치 되어있어야 한다.

2. WSL CUDA Toolkit for WSL 설치 : WSL이 윈도우의 NVIDIA 드라이버와 통신할 수 있도록 해주는 도구.
```
# 이전 CUDA 설치 제거 (혹시 모를 경우)
sudo apt-key del 7fa2af80
sudo apt-get purge cuda*
sudo apt-get autoremove

# CUDA for WSL 저장소 설정 및 설치
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda-repo-wsl-ubuntu-12-5-local_12.5.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-5-local_12.5.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-5-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-5
```

## Miniconda 설치
```
# Miniconda 설치 스크립트 다운로드
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 설치 스크립트 실행
bash Miniconda3-latest-Linux-x86_64.sh
```

## Conda 가상 환경 생성
```
conda create -n gpt-oss-20b python=3.10
conda activate gpt-oss-20b
```

## 핵심 라이브러리 설치
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes
pip install intel-extension-for-transformers
```
- torch : 모델을 실행 가능한 프로그램으로 만들어주는 Pytorch 프레임워크.
- transformers : 모델의 아키텍처 사전 정의 및 추상화된 파이프라인 제공. 허깅 페이스와 연동되어 모델을 간단한 함수처럼 호출할 수 있게 도와주는 인터페이스 역할을 함.
- accelerate : 디바이스 매핑(device mapping) 및 훅(hook)을 통한 모델 로딩 제어. device_map="auto" 옵션을 통해, 모델의 전체 크기와 사용 가능한 하드웨어의 용량을 비교하고, 모델의 각 레이어가 어느 디바이스에 할당될지 동적으로 결정한다. 또한 VRAM에 모든 레이어가 올라가지 않을 경우, 일부 레이어를 CPU RAM이나 디스크에 배치(offload) 후 연산이 필요할 때만 VRAM으로 이동시키고 연산이 종료되면 다시 내보내는 정교한 스케쥴링을 수행한다. 따라서 개발자가 수동적으로 최적화를 하지 않고도 거대 모델을 안정적으로 실행할 수 있도록 돕는다.
- bitsandbytes : 동적 양자화 및 역양자화를 위한 라이브러리. 
  accelerate의 훅이 모델의 가중치를 VRAM에 올리기 전에 `bitsandbytes`가 FP16의 가중치 텐서를 받아 4비트 정수 형태로 변환. 
  GPU는 4비트 연산을 수행할 수 없으므로, 실제 행렬 곱셉 연산이 실행되기 전 `bitsandbytes`가 압축되어있던 가중치를 FP16으로 실시간 역양자화 함. 
  이로 인해 속도 저하를 최소화하면서 VRAM 사용량을 획기적으로 줄이는 효과를 얻음.

기본적으로 `gpt-oss` 모델은 MXFP4로 양자화가 되어있지만, 하드웨어 수준에서 MXFP4 포맷을 처리하는 기능은 50시리즈부터 탑재되어 있음. 
따라서 4070ti는 사용 불가. => bitsandbytes로 시도해보자.

양자화가 기본적으로 된 상태로 배포가 이루어지므로, 역양자화 한 상태로 모델을 불러온 뒤 bitsandbytes 로 양자화를 다시 수행해야한다.
이를 위한 라이브러리가 intel-extension-for-transformers의 Mxfp4Config(dequantize=True)이다.
[참고](https://github.com/huggingface/transformers/issues/39939#issuecomment-3164387479)
