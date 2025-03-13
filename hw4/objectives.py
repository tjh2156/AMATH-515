import numpy as np

def get_schwefel(d):
    def f(x):
        sum = 0
        for xi in x:
            sum += xi*np.sin(np.sqrt(np.abs(xi)))
        return 418.9829*d - sum
    
    def gradf(x):
        return -(np.sin(np.sqrt(np.abs(x))) + x/(2*np.sqrt(np.abs(x)))*np.cos(np.sqrt(np.abs(x))))
    
    def hessf(x):
        diag = np.array(-x*np.sin(np.sqrt(np.abs(x)))/(4*np.abs(x)) + 3*np.cos(np.sqrt(np.abs(x)))/(2*np.sqrt(np.abs(x))) + 3*x*np.cos(np.sqrt(np.abs(x)))/(4*np.sqrt(np.abs(x))))
        return np.diag(diag)
    x0 = -1*np.ones(d)

    return f, gradf, hessf, x0

def get_rosenbrock(d):
    def f(x):
        coupled_term = 100*(x[1:]**2-x[:-1])**2
        diagonal_term = (x  - 1.)**2
        return np.sum(coupled_term) + np.sum(diagonal_term)
    
    def gradf(x):
        grad = 2.0 * (x - 1.0)
        diffs = x[1:]**2 - x[:-1]
        grad[:-1] += -200.0 * diffs
        grad[1:]  += 400.0 * x[1:] * diffs
        return grad
    
    def hessf(x):
        diag_main = np.full(d, 2.0)
        diag_main[:-1] += 200.0
        diag_main[1:] += 1200.0 * x[1:]**2 - 400.0 * x[:-1]
        off_diag = np.zeros(d - 1)
        off_diag = -400.0 * x[1:]
        H = (
            np.diag(diag_main) +
            np.diag(off_diag, k=1) +
            np.diag(off_diag, k=-1)
        )
        return H
    x0 = -1 * np.ones(d)
    
    return f,gradf,hessf,x0


#MNIST Logistic regression
mnist_data = np.load('mnist01.npy',allow_pickle=True)
#
A_lgt = mnist_data[0]
b_lgt = mnist_data[1]
x0 = np.zeros(A_lgt.shape[1])

def get_lgt_obj(lam_lgt):
    # define function, gradient and Hessian
    def lgt_func(x):
        y = A_lgt.dot(x)
        return np.sum(np.log(1.0 + np.exp(y))) - b_lgt.dot(y) + 0.5*lam_lgt*np.sum(x**2)
    #
    def lgt_grad(x):
        y = A_lgt.dot(x)
        z = 1.0/(1.0 + np.exp(-y)) - b_lgt
        return A_lgt.T.dot(z) + lam_lgt*x
    #
    def lgt_hess(x):
        y = A_lgt.dot(x)
        z = np.exp(-y)/(1.0 + np.exp(-y))**2
        return A_lgt.T.dot(np.diag(z).dot(A_lgt)) + lam_lgt*np.eye(x.size)
    
    return lgt_func,lgt_grad,lgt_hess,x0
