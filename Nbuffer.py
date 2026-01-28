import numpy as np
import matplotlib.pyplot as plt

r1 = 0.1
r2 = 0.1
f1 = 0.12
f2 = 0.12
mu1 = 0.7
mu2 = 0.7



def get_Nbuffer_matrix(r1, r2, f1, f2,mu1,mu2,buffer_size):

    matrix = np.array([
        [-0.25,  r2,  r1,  0.00,  0.00,  0.00,  0.00,  0.00],
        [ f2, -0.22,  0.00,  r1,  0.00,  0.00,  0.00,  0.00],
        [ f1,  0.00, -0.72,  r2,  0.00,  0.00,  mu1,  0.00],
        [ 0.00,  f1,  f2, -0.69,  0.00,  0.00,  0.00,  mu1],
        [ 0.00,  0.00,  0.00,  0.00, -0.25,  r2,  r1,  0.00],
        [ 0.00,  mu2,  0.00,  0.00,  f2, -0.92,  0.00,  r1],
        [ 0.00,  0.00,  0.00,  0.00,  f1,  0.00, -0.22,  r2],
        [ 0.00,  0.00,  0.00,  mu2,  0.00,  f1,  f2, -0.89]
    ])


    ring = matrix[0:4,0:4]
    down = matrix[4:,0:4]
    up = matrix[0:4,4:]
    
    nbr_blocks = buffer_size+1

    A = np.zeros((nbr_blocks*4, nbr_blocks*4))


    for i in range(nbr_blocks):
        A[i*4:(i+1)*4, i*4:(i+1)*4] = ring

    for i in range(nbr_blocks-1):
        A[(i+1)*4:(i+2)*4, i*4:(i+1)*4] = down
        A[i*4:(i+1)*4, (i+1)*4:(i+2)*4] = up

    for i in range(A.shape[0]):
        _sum = np.sum(A[i])
        A[i,i] -= _sum
    return A


def getNbuffer_prop_dist(r1,r2,f1,f2,mu1,mu2,buffer_size):
    A = get_Nbuffer_matrix(r1,r2,f1,f2,mu1,mu2,buffer_size)
    M = np.vstack((A.T, np.ones((1,A.shape[0]))))
    b = np.zeros((M.shape[0],1))
    b[-1,0] = 1
    p,_,_,_ = np.linalg.lstsq(M,b, rcond = None)
    p = p.reshape(-1,4)
    p_buffer = [np.sum(row) for row in p]
    return p_buffer
 

p = getNbuffer_prop_dist(r1,r2,f1,f2,mu1,mu2,20)
p_cumulative = np.cumsum(p)
plt.xticks(range(len(p)))
plt.bar(range(len(p)), p, alpha=0.5, label='Probability Density')
plt.bar(range(len(p_cumulative)), p_cumulative, alpha=0.5, label='P(X <= x)')
plt.xlabel('Buffer State')
plt.ylabel('Probability')
plt.title('N-Buffer Probability Distribution')
plt.legend()


alpha = 0.01
threshold = 1 - alpha
index = np.argmax(p_cumulative > threshold)
print(f"First index where F(x)> {threshold}: {index}")
print(f"P(X <= {index}) = {p_cumulative[index]:.4f}")
plt.show()









