from matplotlib import pyplot as plt
import numpy as np

x = np.arange(102400, 512001, 12800);
nonStreamed = np.array([325,306,346,369,388,408,455,469,505,529,550,564,600,656,656,1405,713,726,752,805,834,831,881,889,926,955,1002,995,1026,1039,1104,1105,1135])
streamed = np.array([293, 287,317,322,351,369,386,435,438,480,482,482,513,526,579,563,600,607,655,658,661,714,727,750,767,786,806,843,837,881,889,913,940])

gain = nonStreamed / streamed 

plt.plot(x, gain, ls='--', marker='+', lw=2)
plt.xlabel('Vector length')
plt.ylabel('Performance gain with 4 CUDA streams')
plt.show()
