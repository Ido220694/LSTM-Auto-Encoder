

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    #3.1, part 1
    col = []
    rng = np.random.default_rng()
    rdm = rng.random(size=(10000, 50))
    rdm = rdm - rdm.mean(axis=1, keepdims=True) +0.5
    for i in range(50):
        col.extend([f"d{i:02}"])
    r = pd.DataFrame(rdm, columns=col)
    train_set, test_set = train_test_split(r, test_size=0.4)
    val_set, test_set = train_test_split(test_set, test_size=0.5)
    val_set.to_csv("./data/toy_val.csv", index=False)
    test_set.to_csv("./data/toy_test.csv", index=False)
    train_set.to_csv("./data/toy_train.csv", index=False)

    #3.3, part 1

    sp500 = pd.read_csv("./data/SP 500 Stock Prices 2014-2017.csv", parse_dates=["date"], dtype={"close": np.float32})
    closes =[]
    dates = []

    for i in range(497471):
        if(sp500["symbol"][i] == "AMZN"):
             closes.append(sp500["high"][i])
             dates.append(sp500["date"][i])

    plt.plot( dates, closes)
    plt.show()

    closes =[]
    dates = []

    for i in range(497471):
        if(sp500["symbol"][i] == "AMZN"):
             closes.append(sp500["high"][i])
             dates.append(sp500["date"][i])

    plt.plot( dates, closes)
    plt.show()


