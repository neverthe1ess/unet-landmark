import os
import numpy as np
import matplotlib.pyplot as plt

# 결과 표시하는 코드
result_dir = './result/numpy'

lst_data = os.listdir(result_dir)

lst_label = [f for f in lst_data if f.startswith('label')]
lst_input = [f for f in lst_data if f.startswith('input')]
lst_output = [f for f in lst_data if f.startswith('output')]

lst_label.sort()
lst_input.sort()
lst_output.sort()

for id in range(len(lst_output)):
    label = np.load(os.path.join(result_dir, lst_label[id]))
    input = np.load(os.path.join(result_dir, lst_input[id]))
    output = np.load(os.path.join(result_dir, lst_output[id]))

    plt.figure(figsize=(15, 5))

    ## 원본 이미지
    plt.subplot(131)
    plt.imshow(input, cmap='gray')
    plt.title('input')

    ## 레이블 이미지(정답)
    plt.subplot(132)
    plt.imshow(label, cmap='gray')
    plt.title('label')

    ## 원본 + 예측(output) 오버레이
    plt.subplot(133)
    plt.imshow(input, cmap='gray')  # 원본 input 표시
    plt.imshow(output, cmap='Reds', alpha=0.5)  # output을 빨간색으로 overlay
    plt.title('input + output (Red)')

    plt.show()








