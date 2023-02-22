
import pickle
filename = 'log.pkl'
with open(filename, 'rb') as handle:
    b = pickle.load(handle)

print(b)
print(len(b))