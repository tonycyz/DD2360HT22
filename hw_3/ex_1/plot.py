from matplotlib import pyplot as plt


labels = ['20000', '40000', '60000', '80000', '100000', '200000'] # inputLength

memcpy_HtoD = [36.544, 63.04, 88.831, 112.8, 138.69, 414.52] # CUDA memcpy HtoD (us)
memcpy_DtoH = [14.336, 28.256, 43.071, 60.225, 72.255, 147.58] # CUDA memcpy DtoH (us)
kernel = [7.008, 8.928, 11.328, 12.960, 16.223, 24.831] # __global__ vecAdd

width = 0.35

fig, ax = plt.subplots()

ax.bar(labels, memcpy_HtoD, width, label='memcpy HtoD')
ax.bar(labels, memcpy_DtoH, width, label='memcpy DtoH')
ax.bar(labels, kernel, width, label='kernel')
ax.set_xlabel('Problem size (vector length)')
ax.set_ylabel('Time spent (us)')
ax.set_title('Profiling results of VecAdd')
ax.legend()

plt.show()
