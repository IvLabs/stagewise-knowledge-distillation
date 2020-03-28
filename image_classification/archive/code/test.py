import matplotlib.pyplot as plt

a = [i ** 2 for i in range(10)]
b = range(10)

plt.plot(b, a, 'r', label = 'square')
plt.plot(b, b, 'b', label = 'linear')
plt.legend()
plt.show()