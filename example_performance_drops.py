import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import datetime

start = pd.Timestamp('2020-01-01')
end = pd.Timestamp('2020-01-02')

LEN = 18

# t = np.linspace(start.value, end.value, LEN)
# t = pd.to_datetime(t)
# t = t.strftime("%H")
t = np.linspace(0, 60, LEN)
random.seed(31)

x = np.asarray(t)
print(x)

y = np.zeros(x.size, dtype=np.float32)
y[0] = 0.6
for i in range(1, y.size):
    delta = random.uniform(0.0, 0.15)
    if y[i - 1] + delta > 1.0:
        y[i] = y[i - 1] - delta
    else:
        y[i] = y[i - 1] + delta

    if i % 6 == 0:
        if y[i] - 0.5 < 0.0:
            y[i] = 0.05
        else:
            y[i] = y[i - 1] - 0.5

plt.plot(x, y)
plt.show()
