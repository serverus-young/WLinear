import numpy as np
import matplotlib.pyplot as plt
import pywt

# 获取 db3 小波的低通和高通滤波器系数
wavelet = 'db3'
lowpass, highpass = pywt.Wavelet(wavelet).filter_bank[:2]  # 获取低通和高通滤波器

# 绘制低通滤波器和高通滤波器
plt.figure(figsize=(12, 6))

# 低通滤波器
plt.subplot(1, 2, 1)
plt.plot(lowpass, label='Low-pass Filter (h[n])', color='b')
plt.title(f'{wavelet} - Low-pass Filter')
plt.xlabel('Sample Index')
plt.ylabel('Filter Coefficients')
plt.grid(True)
plt.legend()

# 高通滤波器
plt.subplot(1, 2, 2)
plt.plot(highpass, label='High-pass Filter (g[n])', color='r')
plt.title(f'{wavelet} - High-pass Filter')
plt.xlabel('Sample Index')
plt.ylabel('Filter Coefficients')
plt.grid(True)
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()
