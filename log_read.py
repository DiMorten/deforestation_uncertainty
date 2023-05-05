
import pickle
filename = 'log_evidential_dirichletplot1.pkl'
with open(filename, 'rb') as handle:
    b = pickle.load(handle)

print(b)
print(len(b))