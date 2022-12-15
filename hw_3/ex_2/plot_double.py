from matplotlib import pyplot as plt
import  numpy as np

labels = ['(128, 512, 128)', '(256, 512, 256)', '(512, 512, 512)', '(768, 512, 768)', '(1024, 512, 1024)']

kernel = [264.48, 1067.5, 3652.0, 8269.8, 17930]
kernel = np.array(kernel)
memcpy_HtoD = [91.102, 175.39, 618.96, 1065.7, 1535.6]
memcpy_HtoD = np.array(memcpy_HtoD)
memcpy_DtoH = [12.159, 41.951, 164.48, 2133.2, 4429.6]
memcpy_DtoH = np.array(memcpy_DtoH)
width = 0.3

fig, ax = plt.subplots()

ax.bar(labels, kernel, width, label='kernel')

ax.bar(labels, memcpy_HtoD, width, bottom=kernel, label='memcpy HtoD')

ax.bar(labels, memcpy_DtoH, width, bottom=(kernel+memcpy_HtoD), label='memcpy DtoH')

ax.set_xlabel('Problem definition: (numARows, numBRows, numBColumns)')
ax.set_ylabel('Time spent (us)')
ax.set_title('Profiling results of GEMM (DOUBLE)')
ax.legend()

plt.show()
