import numpy as np
ar = np.array([[1,2],[3,4],[2,3]])
nr = ar/np.linalg.norm(ar, ord=2, axis=0, keepdims=True)
nr = nr.ravel()
#ar.ravel()