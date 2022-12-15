from matplotlib import pyplot as plt
import  numpy as np

labels = ['(128, 512, 128)', '(256, 512, 256)', '(512, 512, 512)', '(768, 512, 768)', '(1024, 512, 1024)']

kernel = [205.69, 762.13, 2576.4, 5772.1, 10218.0]
kernel = np.array(kernel)
memcpy_HtoD = [52.864, 95.198, 179.36, 319.42, 618.71]
memcpy_HtoD = np.array(memcpy_HtoD)
memcpy_DtoH = [6.976, 22.111, 81.63, 776.69, 1505.7]
memcpy_DtoH = np.array(memcpy_DtoH)
width = 0.3

fig, ax = plt.subplots()

ax.bar(labels, kernel, width, label='kernel')

ax.bar(labels, memcpy_HtoD, width, bottom=kernel, label='memcpy HtoD')

ax.bar(labels, memcpy_DtoH, width, bottom=(kernel+memcpy_HtoD), label='memcpy DtoH')

ax.set_xlabel('Problem definition: (numARows, numBRows, numBColumns)')
ax.set_ylabel('Time spent (us)')
ax.set_title('Profiling results of GEMM (FLOAT)')
ax.legend()

plt.show()
