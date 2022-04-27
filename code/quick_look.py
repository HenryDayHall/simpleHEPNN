from matplotlib import pyplot as plt
plt.ion()
import awkward as ak
import data_readers

parts, data = data_readers.read()

for name in data.keys():
    signal = ak.ravel(data[name][0])
    background = ak.ravel(data[name][1])
    plt.hist([signal, background], histtype='step', label=parts, bins=50)
    plt.xlabel(name)
    plt.legend()
    plt.ylabel("counts")
    input("Hit enter")
    plt.close()


