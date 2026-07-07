import math
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
import torch.nn.functional as F
import pandas as pd
import numpy as np
            

# ==========================================
# [🔥 외부 조종 스위치] trainer.py에서 변경할 변수들
# ==========================================
GLOBAL_ADC_BIT = 4
GLOBAL_ADC_CUTOFF = 0.5
GLOBAL_ADC_BYPASS = False  # 🚀 [추가] ADC 완전 무시 스위치
GLOBAL_ENABLE_LUT_NOISE = False

# class fake_quant(torch.autograd.Function): # weight_f -> weight_q로 quantization 
#     @staticmethod
#     def forward(ctx, weight_f, quant_bit=8):
#         weight_max = torch.max(weight_f).cuda()
#         weight_min = torch.min(weight_f).cuda()
#         val_max = (weight_max-weight_min).abs() # weight range
#         s = (val_max) / (2**(quant_bit)-0.5)
#         weight_scaled = (weight_f) / s   # scale로 나누기
#         weight_q = torch.round(weight_scaled) # quantization
#         return weight_q 


#     @staticmethod   # gradient는 그대로 전달
#     def backward(ctx, grad_outputs):
#         return grad_outputs, None



# class fake_quant_s(torch.autograd.Function): #  fake_qaunt에서 계산한 것 중 s(scale 값) 만 return
#     @staticmethod # mself 안쓰게 하고 싶을 때 사용 단, self에 접근 불가
#     def forward(ctx, weight_f, quant_bit=8):
#         weight_max = torch.max(weight_f).cuda()
#         weight_min = torch.min(weight_f).cuda()
#         #val_max = torch.max(weight_max.abs(), weight_min.abs())
#         val_max = (weight_max-weight_min).abs()
#         s = (val_max) / (2**(quant_bit)-0.5)
#         return s


#     @staticmethod
#     def backward(ctx, grad_outputs):
#         return None, None # no traning for scale 계산은 하지만 업데이트는 안함 
 
# class FakeQuant(nn.Module): 
#     def __init__(self, quant_bit=8):
#         super(FakeQuant, self).__init__()
#         self.quant_bit = quant_bit


#     def forward(self, weight_f):
#         weight_q = fake_quant.apply(weight_f, self.quant_bit)
#         return weight_q
    
    
    
# class FakeQuant_s(nn.Module): # wrapper  class for fake_quant_s 
#     # 굳이 또 만든 이유: nn.Module 형태로 만들기 위해 -> 실제 layer 내에서 사용하려고
#     def __init__(self, quant_bit=8):
#         super(FakeQuant_s, self).__init__()
#         self.quant_bit = quant_bit


#     def forward(self, weight_f):
#         weight_s = fake_quant_s.apply(weight_f, self.quant_bit)
#         return weight_s   
    

# class QConv2d(torch.nn.Conv2d): # nn.Conv2d 상속 -> 따라서 커널 생성은 super()로 처리
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
#                  first_layer=False,
#                  quant_bit=8,
#                  quant_bit_weight=8): # shape parameter 생성 
#         super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.first_layer = first_layer
#         self.quant_bit = quant_bit
#         self.quant_bit_weight = quant_bit_weight
#         self.padding_mode = padding_mode
#         #self.first_quantizer = FakeQuant(quant_bit=self.quant_bit_weight)
#         #self.first_quantizer_s = FakeQuant_s(quant_bit=self.quant_bit_weight)
#         self.act_quantizer    = FakeQuant(quant_bit=self.quant_bit)
#         self.weight_quantizer = FakeQuant(quant_bit=self.quant_bit_weight)
#         self.act_quantizer_s  = FakeQuant_s(quant_bit=self.quant_bit)
#         self.weight_quantizer_s = FakeQuant_s(quant_bit=self.quant_bit_weight)
        
#         self.activation_outputs = []
#         self.weight_outputs = []
        
#         if bias:
#             self.bias = torch.nn.Parameter(torch.zeros(out_channels))
#         else:
#             self.bias = None    
    
#     def save_for_statistics(self, x_q, weight_q): # 추후 분석 용도로 저장 (list에 append)
#         self.activation_outputs.append(x_q.detach().cpu())
#         self.weight_outputs.append(weight_q.detach().cpu())
        
#     def forward(self, x):
#         # Quantization
#         #if self.first_layer:
#         #    x_q = self.first_quantizer(x)  
#         #    x_s = self.first_quantizer_s(x)
#         #else:
#         #    x_q = self.act_quantizer(x)
#         #    x_s = self.act_quantizer_s(x)

#         weight_q = self.weight_quantizer(self.weight)   # scale 안곱해져있음 weight quantized
#         x_q = self.act_quantizer(x)                     # scale 안곱해져있음 x quantized
#         weight_s = self.weight_quantizer_s(self.weight) # scale 값 (weight_f -> s)
#         x_s = self.act_quantizer_s(x)                   # scale 값 (x_f -> s)
        
#         if self.first_layer:
#             self.mode = 'inference'
        
#         y = F.conv2d(x_q*x_s, weight_q*weight_s, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) # scale 곱해진 상태로 conv 연산
#         # bias는 quantization 안함
        
#         return y 
    
# class QLinear(torch.nn.Linear):
    # def __init__(self, in_features, out_features, bias=True, quant_bit=8, quant_bit_weight = 8):
        # super(QLinear, self).__init__(in_features, out_features, bias)
        # self.quant_bit = quant_bit
        # self.quant_bit_weight = quant_bit_weight
        # self.act_quantizer    = FakeQuant(quant_bit=self.quant_bit)
        # self.weight_quantizer = FakeQuant(quant_bit=self.quant_bit_weight)
        
        
    # def forward(self, x): # Linear layer에서는 scale 곱 안함 
        # weight_q = self.weight_quantizer(self.weight)
        # bias_q = self.weight_quantizer(self.bias)
        # x_q = self.act_quantizer(x)
        # y = F.linear(x_q, weight_q, bias=bias_q)
        # return y


# ==========================================
# [🔥 추가] 학습형 활성화 양자화 (PACT)
# ==========================================
class PACT_Quant(nn.Module):
    def __init__(self, quant_bit=4):
        super(PACT_Quant, self).__init__()
        self.quant_bit = quant_bit
        # 초기화 안 됨 상태(-1.0)로 시작
        self.alpha = nn.Parameter(torch.tensor(-1.0)) 

    def forward(self, x):
        # 1. 자동 캘리브레이션 (첫 배치 기준 99.9%로 초기화)
        if self.alpha.data < 0:
            with torch.no_grad():
                init_val = torch.quantile(x.detach().abs(), 0.999).clamp(min=1e-4)
                self.alpha.data.fill_(init_val)

        # 2. PACT 클리핑 (torch.minimum을 써야 alpha가 학습됨!)
        x_relu = F.relu(x)
        x_clipped = torch.minimum(x_relu, self.alpha)
        
        # 3. 스케일 계산
        levels = (1 << self.quant_bit) - 1
        s = self.alpha / levels
        
        # 4. 순수 정수화 및 STE(미분 고속도로) 적용
        x_scaled = x_clipped / s
        x_q = torch.round(x_scaled)
        # 라운딩(round)은 미분이 안 되므로, 역전파 통과를 위한 파이토치 트릭
        x_q_ste = (x_q - x_scaled).detach() + x_scaled
        
        return x_q_ste # 하드웨어 연산을 위해 순수 '정수' 반환

class PACT_Quant_s(nn.Module):
    def __init__(self, pact_module):
        super(PACT_Quant_s, self).__init__()
        self.pact = pact_module # 짝꿍 PACT 모듈의 alpha를 공유함

    def forward(self, x):
        levels = (1 << self.pact.quant_bit) - 1
        s = self.pact.alpha / levels
        return s # 스케일만 반환


# ==========================================
# 1. 하드웨어 ADC 노이즈 모사 클래스
# ==========================================
class ADC_quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, psum, adc_bits=4, dequant=True, fs=240.0, cutoff_ratio=0.5):
        if cutoff_ratio == 0.0:
            cutoff_max = fs
        else:
            cutoff_max = (256.0 * (1.0 - cutoff_ratio)) # 논문 스펙 기준 256 기준 Cut-off

        # Cut-off 적용 (128 이상 잘라내기)
        psum_cutoff = torch.clamp(psum, min=0.0, max=(cutoff_max - 1.0))
        
        levels = (1 << adc_bits)
        scale = cutoff_max / levels

        # Quantization
        psum_q = torch.floor(psum_cutoff / scale)
        psum_q = torch.clamp(psum_q, 0.0, float(levels-1))

        if dequant:
            return psum_q * scale
        return psum_q

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None, None, None, None


# ==========================================
# 2. Signed / Unsigned 지원 양자화 클래스
# ==========================================
class fake_quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_f, quant_bit=8, is_unsigned=False):
        if is_unsigned:
            # Unsigned 양자화 [0, 2^b - 1] - ReLU 이후 입력 데이터용
            val_max = torch.max(x_f).clamp(min=1e-8)
            s = val_max / ((1 << quant_bit) - 1)
            x_q = torch.round(x_f / s)
            x_q = torch.clamp(x_q, 0, (1 << quant_bit) - 1)
        else:
            # Signed 대칭 양자화 [-2^(b-1), 2^(b-1) - 1] - Weight 및 첫 레이어 입력용
            val_max = torch.max(x_f.abs()).clamp(min=1e-8)
            s = val_max / ((1 << (quant_bit - 1)) - 1)
            x_q = torch.round(x_f / s)
            x_q = torch.clamp(x_q, -(1 << (quant_bit - 1)), (1 << (quant_bit - 1)) - 1)
        
        # [🔥 추가 1] 역전파에서 사용하기 위해 계산된 스케일(s)을 저장
        ctx.save_for_backward(s)

        return x_q

    @staticmethod
    def backward(ctx, grad_outputs):
        # [🔥 추가 2] 저장해둔 스케일(s)을 불러오기
        s, = ctx.saved_tensors
        
        # [🔥 추가 3] 작아진 기울기를 다시 s로 나누어 원상 복구!
        return grad_outputs / s, None, None

class fake_quant_s(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_f, quant_bit=8, is_unsigned=False):
        if is_unsigned:
            val_max = torch.max(x_f).clamp(min=1e-8)
            s = val_max / ((1 << quant_bit) - 1)
        else:
            val_max = torch.max(x_f.abs()).clamp(min=1e-8)
            s = val_max / ((1 << (quant_bit - 1)) - 1)
        return s

    @staticmethod
    def backward(ctx, grad_outputs):
        return None, None, None

class FakeQuant(nn.Module): 
    def __init__(self, quant_bit=8, is_unsigned=False):
        super(FakeQuant, self).__init__()
        self.quant_bit = quant_bit
        self.is_unsigned = is_unsigned

    def forward(self, x_f):
        return fake_quant.apply(x_f, self.quant_bit, self.is_unsigned)
        
class FakeQuant_s(nn.Module): 
    def __init__(self, quant_bit=8, is_unsigned=False):
        super(FakeQuant_s, self).__init__()
        self.quant_bit = quant_bit
        self.is_unsigned = is_unsigned

    def forward(self, x_f):
        return fake_quant_s.apply(x_f, self.quant_bit, self.is_unsigned)


# ==========================================
# 3. 통합 QConv2d (소프트웨어 및 CIM 하드웨어 모사)
# ==========================================
class QConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 first_layer=False, quant_bit=4, quant_bit_weight=8): 
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        
        self.first_layer = first_layer
        
        # [🔥 추가] forward 함수에서 선생님(32)인지 학생(4)인지 구분할 수 있도록 변수 저장!
        self.quant_bit = quant_bit

        # 1. 가중치(Weight)는 항상 Signed 8-bit 입니다.
        self.weight_quantizer = FakeQuant(quant_bit=quant_bit_weight, is_unsigned=False)
        self.weight_quantizer_s = FakeQuant_s(quant_bit=quant_bit_weight, is_unsigned=False)
        
        # if self.first_layer:
        #     # 2. 첫 번째 레이어: 입력이 Normalized 되어 음수가 존재하므로 Signed 8-bit 처리
        #     self.act_quantizer = FakeQuant(quant_bit=8, is_unsigned=False)
        #     self.act_quantizer_s = FakeQuant_s(quant_bit=8, is_unsigned=False)
        # else:
        #     # 3. 이후 레이어: ReLU 통과 후 값이므로 Unsigned 4-bit (메모 기준 4비트 처리)
        #     self.act_quantizer = FakeQuant(quant_bit=quant_bit, is_unsigned=True)
        #     self.act_quantizer_s = FakeQuant_s(quant_bit=quant_bit, is_unsigned=True)
            
        if self.first_layer:
            # 첫 레이어는 8-bit 그대로 유지
            self.act_quantizer = FakeQuant(quant_bit=8, is_unsigned=False)
            self.act_quantizer_s = FakeQuant_s(quant_bit=8, is_unsigned=False)
        else:
            # 🚀 [수정] 4-bit 양자화에는 학습하는 PACT 가위 장착!
            self.act_quantizer = PACT_Quant(quant_bit=quant_bit)
            self.act_quantizer_s = PACT_Quant_s(self.act_quantizer) # 동일한 모듈을 넘겨줘서 alpha 공유

        self.activation_outputs = []
        self.weight_outputs = []

        # [히스토그램 저장용 텐서 추가] 0부터 255까지의 발생 빈도를 누적할 텐서
        self.register_buffer('mac_hist', torch.zeros(256))

        self.register_buffer('comp_offset_base', torch.randn(16))


        # [LUT 초기화] 16개의 ABL 라인, 17개의 MAC 스텝 (0~240, step 15)
        mu_lut = torch.zeros((16, 17))
        sigma_lut = torch.zeros((16, 17))
        
        try:
  
            # 1. CSV 파일 로드 및 하단 요약 행 필터링
            df = pd.read_csv('W_AMU_16X1_MAC.mt0.csv', skiprows=2)
            df.columns = [col.strip() for col  in df.columns]
            is_mc_sample = pd.to_numeric(df.iloc[:, 0], errors='coerce').notna()
            df_clean = df[is_mc_sample].copy()
            
            # =========================================================
            # 🔥 핵심 수정: 하드웨어 스펙 동적 추출 (오류 수정)
            # =========================================================
            # 컴프리헨션 대신 일반적인 리스트 필터링 방식으로 안전하게 추출
            columns_list = df_clean.columns.tolist()
            mac_0_cols = []
            mac_240_cols = []
            for column_name in columns_list:
                if column_name.startswith('mac_000_abl_'):
                    mac_0_cols.append(column_name)
                elif column_name.startswith('mac_240_abl_'):
                    mac_240_cols.append(column_name)
            # 100번의 MC 결과 전체를 평균 내어 "회로의 물리적 기준점" 도출 (mV 변환)
            mac_0_data = pd.to_numeric(df_clean[mac_0_cols].values.flatten(), errors='coerce')
            mac_240_data = pd.to_numeric(df_clean[mac_240_cols].values.flatten(), errors='coerce')
            
            V_ideal_start = np.nanmean(mac_0_data) * 1000.0
            V_end = np.nanmean(mac_240_data) * 1000.0
            
            # 1 MAC 스텝당 전압 하강폭 동적 계산
            VOLTAGE_PER_MAC = (V_ideal_start - V_end) / 240.0

            self.VOLTAGE_PER_MAC = VOLTAGE_PER_MAC
            
            # (디버깅용 출력: VDD를 바꾸거나 온도를 바꿔서 CSV를 뽑아도 스스로 적응함을 확인 가능)
            # print(f"HW Specs Loaded -> V_start: {V_ideal_start:.2f}mV, 1 MAC Step: {VOLTAGE_PER_MAC:.2f}mV")
            # =========================================================
            
            # 2. 데이터 추출 및 LUT 맵핑
            for m_idx, mac in enumerate(range(0, 241, 15)):
                prefix = f'mac_{mac:03d}_abl_'
                for abl_idx in range(16):
                    col_name = f'{prefix}{abl_idx}'
                    
                    if col_name in df_clean.columns:
                        data_mv = pd.to_numeric(df_clean[col_name], errors='coerce').dropna().values * 1000.0
                        
                        # Std Dev 변환 (mV -> MAC scale)
                        std_mv = np.std(data_mv)
                        sigma_lut[abl_idx, m_idx] = std_mv / VOLTAGE_PER_MAC
                        
                        # Mean Offset 변환 (mV -> MAC scale)
                        ideal_v = V_ideal_start - (mac * VOLTAGE_PER_MAC)
                        mean_v = np.mean(data_mv)
                        
                        mu_offset_mv = ideal_v - mean_v 
                        mu_lut[abl_idx, m_idx] = mu_offset_mv / VOLTAGE_PER_MAC
                        
        except Exception as e:
            print(f"LUT 로드 실패 (기본값 0 적용): {e}")

        # GPU 할당을 위한 Buffer 등록
        self.register_buffer('hw_mu_lut', mu_lut)
        self.register_buffer('hw_sigma_lut', sigma_lut)

    def save_for_statistics(self, x_q, weight_q): 
        self.activation_outputs.append(x_q.detach().cpu())
        self.weight_outputs.append(weight_q.detach().cpu())

    def _to_int_weights(self, weight_q):
        # 2의 보수 모사를 위해 Signed 정수 [-128, 127]를 Unsigned 8-bit 패턴 [0, 255]으로 변환
        W_u = torch.where(weight_q < 0, weight_q + 256, weight_q).to(torch.int32)
        return W_u

    def forward(self, x):

        # ==========================================================
        # [🔥 추가] 선생님 모델(32-bit) 프리패스 스위치
        # 양자화 및 CIM 모사를 완벽하게 건너뛰고 순정 Float32 연산 수행
        # ==========================================================
        if self.quant_bit == 32:
            return F.conv2d(x, self.weight, bias=self.bias, 
                            stride=self.stride, padding=self.padding, 
                            dilation=self.dilation, groups=self.groups)

        # ----------------------------------------------------------
        # 아래부터는 기존 학생용(4-bit) 양자화 및 CIM 모사 로직 그대로 유지

        # 가중치 및 입력 양자화 연산
        weight_q = self.weight_quantizer(self.weight) 
        weight_s = self.weight_quantizer_s(self.weight) 
        
        x_q = self.act_quantizer(x) 
        x_s = self.act_quantizer_s(x)
        
        # [슬라이드 로직 분기]
        if self.first_layer:
            # 첫 번째 레이어: CIM 연산을 거치지 않고 소프트웨어 연산(F.conv2d) 그대로 진행
            y = F.conv2d(x_q * x_s, weight_q * weight_s, bias=self.bias, 
                         stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            return y
            
        else:
            # 이후 레이어: CIM 하드웨어 모사 진행 (Unfold -> 타일 분할 -> Bit-serial -> ADC 양자화 -> Fold)
            W_u = self._to_int_weights(weight_q)
            
            # --- Unfold (공간 연산을 텐서 곱으로 변환) ---
            N, Cin, H, W = x_q.shape
            Cout = self.out_channels
            
            H_out = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
            W_out = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
            L = H_out * W_out 
            taps = self.kernel_size[0] * self.kernel_size[1]

            x_unfold = F.unfold(x_q, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
            x_unfold = x_unfold.view(N, Cin, taps, L).permute(0, 3, 2, 1) 

            # --- Channel Tiling (16 Row 하드웨어 제약) ---
            cin_tile = 16
            T = (Cin + cin_tile - 1) // cin_tile
            Cin_pad = T * cin_tile
            
            if Cin != Cin_pad:
                pad_tensor = torch.zeros((N, L, taps, Cin_pad - Cin), device=x.device, dtype=x_unfold.dtype)
                x_unfold = torch.cat([x_unfold, pad_tensor], dim=-1)
            X_ = x_unfold.view(N, L, taps, T, cin_tile)

            # W_u_pad = W_u.view(Cout, taps, Cin)
            # if Cin != Cin_pad:
            #     w_pad_tensor = torch.zeros((Cout, taps, Cin_pad - Cin), device=x.device, dtype=W_u_pad.dtype)
            #     W_u_pad = torch.cat([W_u_pad, w_pad_tensor], dim=-1)
            # W_u_pad = W_u_pad.view(Cout, taps, T, cin_tile)

            # ==========================================
            # [여기서부터 수정] Weight Tiling 차원 재배열 오류 수정
            # ==========================================
            # 1. 공간(kH, kW)을 taps로 묶은 뒤, 축을 안전하게 교환(permute)
            W_u_reshaped = W_u.view(Cout, Cin, taps).permute(0, 2, 1) # 결과: (Cout, taps, Cin)
            
            # 2. 패딩 적용
            if Cin != Cin_pad:
                w_pad_tensor = torch.zeros((Cout, taps, Cin_pad - Cin), device=x.device, dtype=W_u_reshaped.dtype)
                W_u_pad = torch.cat([W_u_reshaped, w_pad_tensor], dim=-1)
            else:
                W_u_pad = W_u_reshaped
            
            # 3. permute 후에는 메모리가 연속적(contiguous)이지 않으므로 반드시 contiguous() 호출
            W_u_pad = W_u_pad.contiguous().view(Cout, taps, T, cin_tile)

            # --- Bit-serial 연산 및 ADC 통과 ---
            psum_bitplanes = torch.zeros((N, L, taps, T, Cout), device=x.device)


            for b in range(8):
                # 1-bit 가중치 추출
                w_b = ((W_u_pad >> b) & 1).to(dtype=X_.dtype)

                # 16채널 단위 부분합 도출 (아날로그 누적 모사)
                psum_b = torch.einsum('nltic,otic->nltio', X_, w_b)


                # ==========================================================
                # [🔥 핵심 1] Weight Bit-Parallel 기반 ABL 물리적 위치 추적
                # ==========================================================
                if not self.training: 
                    # 1. 출력 채널(Cout) 인덱스 배열 생성 [0, 1, 2, ..., Cout-1]
                    o_indices = torch.arange(Cout, device=X_.device)
                    
                    # 2. 선재님의 AMU 맵핑 공식 적용: ABL 번호 = ((o * 8) + b) % 16
                    abl_indices = ((o_indices * 8) + b) % 16 
                    
                    # ==========================================================
                    # [🔥 핵심 2] 현재 MAC 값(0~240)을 LUT 열(Col) 인덱스(0~16)로 변환
                    # ==========================================================
                    # psum_b는 현재 0~240 사이의 아날로그 누적 값입니다.
                    # 이를 15로 나누고 반올림하여 0~16 사이의 정수 인덱스로 만듭니다.
                    mac_step_idx = torch.clamp(torch.round(psum_b / 15.0), 0, 16).long()
                    
                    # ==========================================================
                    # [🔥 핵심 3] 5차원 텐서 브로드캐스팅 및 노이즈 맵핑 (루프 없음!)
                    # ==========================================================
                    # abl_indices는 1차원(Cout 크기)이므로, 이를 psum_b(5차원) 모양에 맞게 확장합니다.
                    abl_bcast = abl_indices.view(1, 1, 1, 1, Cout).expand_as(mac_step_idx)
                    
                    # 마법의 2D 맵핑: [ABL 인덱스, MAC 스텝 인덱스]로 LUT를 한 번에 조회합니다.
                    # 결과물인 mu_val과 sigma_val은 psum_b와 동일한 (N, L, taps, T, Cout) 크기가 됩니다.
                    # mu_val = self.hw_mu_lut[abl_bcast, mac_step_idx]
                    mu_val = 0.0
                    sigma_val = self.hw_sigma_lut[abl_bcast, mac_step_idx]
                    
                    # 정규 분포 노이즈 생성 및 주입 (이때 텐서 크기가 완벽히 일치하여 1:1로 더해짐)
  

                    global GLOBAL_ENABLE_LUT_NOISE
                    if GLOBAL_ENABLE_LUT_NOISE:
                        noise = torch.normal(mean=mu_val, std=sigma_val)
                        psum_b = psum_b + noise 
                        

                        # ==========================================================
                        # 🔥 2. 동적 비교기(Dynamic Comparator) 오프셋 주입
                        # ==========================================================
                        # HSPICE 몬테카를로 시뮬레이션(28nm)에서 직접 추출한 1 Sigma 값 (mV)
                        COMP_OFFSET_MV = 19.41 
                        
                        # 파이토치의 MAC 스케일(단위)로 변환
                        comp_sigma = COMP_OFFSET_MV / self.VOLTAGE_PER_MAC
                        
                        # [🚀 핵심 수정] 매번 튀는 랜덤 노이즈(열잡음)가 아닌 물리적 고정 오프셋 적용!
                        # __init__에서 생성한 16개 비교기의 고유한 베이스 노이즈에 현재 스케일(comp_sigma)을 곱함
                        fixed_comp_noise = self.comp_offset_base * comp_sigma
                        
                        # 현재 연산이 매핑된 물리적 ABL 번호(abl_bcast)를 인덱스로 사용하여,
                        # 해당 비교기가 가지고 있는 고유한 오프셋 값을 그대로 가져옵니다.
                        comp_noise = fixed_comp_noise[abl_bcast]
                        
                        # ADC 양자화 통과 직전의 아날로그 전압에 고정 비교기 노이즈 최종 합산
                        psum_b = psum_b + comp_noise
                        # ==========================================================


                # ==========================================================
                # [디버깅: MAC 분포 히스토그램 누적]
                if not self.training:
                    with torch.no_grad():
                        # 값을 0~255 사이 정수로 변환 후 빈도수 카운트
                        psum_int = torch.clamp(torch.round(psum_b), 0, 255).long()
                        # bincount를 이용해 빠르고 안전하게 빈도수 더하기
                        self.mac_hist += torch.bincount(psum_int.flatten(), minlength=256)
                # ==========================================================

#                (기존 코드)
                # psum_b = ADC_quant.apply(psum_b, 4, True, 240.0, 0.5)

                global GLOBAL_ADC_BIT, GLOBAL_ADC_CUTOFF, GLOBAL_ADC_BYPASS
                
                if GLOBAL_ADC_BYPASS:
                    # 🚀 [No ADC (Base)] 계단식 양자화(floor)는 무시하되, 
                    # 자르는 기준점은 ADC_quant와 수학적으로 100% 동일하게 맞춤!
                    if GLOBAL_ADC_CUTOFF == 0.0:
                        cutoff_max = 240.0
                    else:
                        cutoff_max = (256.0 * (1.0 - GLOBAL_ADC_CUTOFF))
                    
                    # ADC_quant 클래스와 완벽하게 동일한 조건의 아날로그 포화(Saturation) 모사
                    psum_b = torch.clamp(psum_b, min=0.0, max=(cutoff_max - 1.0))
                    
                else:
                    # 🚀 [실제 HW] Cut-off 적용 + 4-bit 디지털 변환
                    psum_b = ADC_quant.apply(psum_b, GLOBAL_ADC_BIT, True, 240.0, GLOBAL_ADC_CUTOFF)

                # 2's complement Shift and Add
                if b == 7: 
                    psum_bitplanes -= (psum_b * (1 << b))
                else: 
                    psum_bitplanes += (psum_b * (1 << b))

            # --- Fold 및 최종 Scale 복원 ---
            psum_t = psum_bitplanes.sum(dim=3) 
            y_blk = psum_t.sum(dim=2)          
            
            # (N, Cout, H_out, W_out) 형태로 되돌림
            y_blk = y_blk.permute(0, 2, 1).contiguous().view(N, Cout, H_out, W_out)
            
            # # Dequantization: 연산 완료 후 분리해두었던 스케일을 다시 곱해줌
            # out = y_blk * (x_s * weight_s)
            
            # if self.bias is not None:
            #     out += self.bias.view(1, -1, 1, 1)

            # # ==========================================================
            # # [디버깅 로직 추가] F.conv2d 결과와 CIM 모사 결과 비교
            # # ==========================================================
            # with torch.no_grad(): # 메모리 방지
            #     # 1. 원본 소프트웨어 연산 결과 생성
            #     y_baseline = F.conv2d(x_q * x_s, weight_q * weight_s, bias=self.bias, 
            #                           stride=self.stride, padding=self.padding, 
            #                           dilation=self.dilation, groups=self.groups)
                
            #     # 2. 오차율 계산
            #     diff = torch.max(torch.abs(y_baseline - out)).item()
                
            #     # 3. 레이어당 한 번만 출력하도록 체크
            #     if not hasattr(self, 'debug_printed'):
            #         print(f"Layer [{self.in_channels}->{self.out_channels}] | Max Diff: {diff:.4f} | ADC Mean: {torch.mean(y_blk).item():.4f}")
            #         self.debug_printed = True
            # # ==========================================================

            # --- (기존: ADC 통과 및 2의 보수 누적 연산 등 하드웨어 모사 완료) ---
            # 하드웨어 연산의 최종 결과물을 out_hw 로 지정 (이름은 기존 out 변수를 쓰셔도 됩니다)
            out_hw = y_blk * (x_s * weight_s)
            
            if self.bias is not None:
                out_hw += self.bias.view(1, -1, 1, 1)

            # ==========================================================
            # [🔥 추가할 핵심 코드] STE Bypass (미분 고속도로 개통)
            # ==========================================================
            # 1. 미분 가능한 소프트웨어 연산(F.conv2d) 결과를 구합니다. 
            # (이 코드는 예전에 디버깅할 때 쓰셨던 그 코드와 동일합니다)
            out_sw = F.conv2d(x_q * x_s, weight_q * weight_s, bias=self.bias, 
                              stride=self.stride, padding=self.padding, 
                              dilation=self.dilation, groups=self.groups)
            
            # 2. 마법의 STE 수식: Forward는 out_hw가 되고, Backward는 out_sw를 따라갑니다.
            out_final = out_sw + (out_hw - out_sw).detach()
            # ==========================================================

            return out_final

