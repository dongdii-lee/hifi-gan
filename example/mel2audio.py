import os
import sys
import numpy as np
import torch
from PIL import Image
import soundfile as sf

# 1. HiFi-GAN 경로를 시스템 경로에 추가합니다.
sys.path.append('/Users/dongdii/Documents/Visual Studio Code/Kitel/2024/HiFi-GAN/output')  # 'path_to_hifi_gan'를 실제 HiFi-GAN 저장소 경로로 변경하세요.

from models import Generator  # HiFi-GAN의 Generator 모델 임포트

# 2. mel 스펙트로그램 이미지 로드
mel_image = Image.open('/Users/dongdii/Documents/Visual Studio Code/Kitel/2024/librosa/output/Mel-Spectrogram example.png').convert('L')  # 그레이스케일로 변환

# 3. 이미지 크기를 mel 채널 수(80)에 맞게 조정
mel_image = mel_image.resize((mel_image.width, 80))
mel_array = np.array(mel_image).astype(np.float32) / 255.0  # 픽셀 값을 [0, 1]로 정규화
mel_array = mel_array[np.newaxis, :, :]  # 형태를 (1, 80, T)로 변경

# 4. 스케일 되돌리기 (원래 mel 값 범위를 [-4, 4]로 가정)
mel_array = mel_array * 8.0 - 4.0  # [-4, 4] 범위로 변환

# 5. Tensor로 변환
mel_tensor = torch.from_numpy(mel_array)

# 6. HiFi-GAN Generator 모델 로드
generator = Generator()
checkpoint = torch.load('generator_v1.pth.tar', map_location=torch.device('mps')) # 디바이스를 M1 Mac 환경에 맞도록 cpu 에서 mps 로 수정
generator.load_state_dict(checkpoint['generator'])
generator.eval()
generator.remove_weight_norm()

# 7. 오디오 생성
with torch.no_grad():
    audio = generator(mel_tensor)

audio = audio.squeeze().cpu().numpy()

# 8. 웨이브 파일로 저장
sf.write('/Users/dongdii/Documents/Visual Studio Code/Kitel/2024/HiFi-GAN/output/output.wav', audio, samplerate=22050)
