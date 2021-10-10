import numpy as np

def edit_Distance(X, Y):
    Xlen, Ylen = len(X), len(Y)
    table = np.zeros( (Xlen+1, Ylen+1), dtype=int)

    for i in range(1, Xlen+1):
        table[i, 0] = table[i-1, 0] + 1  # del cost - 1
    
    for j in range(1, Ylen+1):
        table[0, j] = table[0, j-1] + 1  #ins-cose - 1
    
    # Recurrence relation
    for i in range(1, Xlen+1):
        for j in range(1, Ylen+1):
            table[i, j] = min(
                table[i-1, j] + 1,
                table[i, j-1] + 1,
                table[i-1, j-1] if (X[i-1]==Y[j-1]) else table[i-1, j-1]+2
            )
    print(table)
    return table[Xlen, Ylen]


source = "text"
target = "test"
print(f'Distance between {source} and {target} is {edit_Distance(source, target)}')