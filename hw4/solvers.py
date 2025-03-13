import numpy as np
import time

#Throughout this code,you may want to modify the structure of these functions to 
#include additional parameters such as stopping tolerance or maximum iterations!

def bisection_linesearch(x,d,f,gradf,c0,c1,t0=1., max_iter = 1000):
    """

    Parameters
    ----------
    x : Previous iterate
    d : Search direction
    f : Objective function
    gradf : Gradient of objective function
    c0 : Parameter for W1 condition (you may want to set a default)
    c1 : Parameter for W2 condition (you may want to set a default)
    t0 : Initial stepsize, by default 1.

    Returns
    -------
    alpha: accepted line search stepsize
    fval: function value at location of accepted step
    g: gradient at the location of accepted step
    
    Note: Returning fval and g can possibly save on extra computation (why!)
    """
    #Your code here
    a = 0
    b = float('inf')
    t = t0

    newX = x + t*d
    wolfe1 = (f(newX) - f(x))
    w1_denom = (t*np.dot(d, gradf(x)))
    wolfe2 = np.dot(gradf(newX), d)
    w2_denom = np.dot(gradf(x),d)
    iter = 0

    while (wolfe1 > c0*w1_denom or wolfe2 < c1*w2_denom ) and iter < max_iter:
        if wolfe1 > c0*w1_denom:
            b = t
            t = (b+a)/2
        elif wolfe2 < c1*w2_denom:
            a = t
            t = min(2*a, (a+b)/2)
        newX = x + t*d
        wolfe1 = (f(newX) - f(x))
        w1_denom = (t*np.dot(d, gradf(x)))
        wolfe2 = np.dot(gradf(newX), d)
        w2_denom = np.dot(gradf(x),d)
        iter += 1

    if iter >= max_iter :
        print("bisection linesearch reached maximum iterations")

    alpha = t
    fval = f(newX)
    g = gradf(newX)
    return alpha,fval,g

def steepest_descent(x0,f,gradf,c0=.5,c1=.5,t0=1., TOL = 1e-6):
    """Steepest descent with weak Wolfe line bisection search

    Parameters
    ----------
    x0 : initialization
    f : objective function
    gradf : gradient of f
    c0 : Parameter for W1 condition (you may want to set a default)
    c1 : Parameter for W2 condition (you may want to set a default)
    t0 : Initial stepsize for line search
    TOL : Tolerance the error must be under
    """

    tol = TOL
    x = x0
    err = np.linalg.norm(gradf(x))
    err_his = []
    func_his = []
    time_his = []
    IDENT = np.eye(len(x))
    
    #record history
    err_his.append(err)
    func_his.append(f(x))

    start = time.time()
    while err >= tol:
        #set up
        grad_f_at_x = gradf(x)
        B = IDENT
        d = -np.dot(B, grad_f_at_x)
        step_size, fval, g = bisection_linesearch(x, d, f, gradf, c0, c1, t0)
        
        #update values
        x = x + step_size * d
        err = np.linalg.norm(g)
        
        #record history
        err_his.append(err)
        func_his.append(fval)
        curr_time = time.time()
        time_his.append(curr_time - start)
        

    return x, func_his, time_his, err_his

def DFP(x0,f,gradf,c0,c1,t0, TOL):
    """DFP with weak Wolfe line bisection search

    Parameters
    ----------
    x0 : initialization
    f : objective function
    gradf : gradient of f
    c0 : Parameter for W1 condition (you may want to set a default)
    c1 : Parameter for W2 condition (you may want to set a default)
    t0 : Initial stepsize for line search (you may want to set a default)
    """
    tol = TOL
    x = x0
    err = np.linalg.norm(gradf(x))
    err_his = []
    func_his = []
    time_his = []
    IDENT = np.eye(len(x))
    B = IDENT

    #record history
    err_his.append(err)
    func_his.append(f(x))

    start = time.time()
    while err >= tol:
        #set up
        grad_f_at_x = gradf(x)
        d = -np.dot(B, grad_f_at_x)
        step_size, fval, g = bisection_linesearch(x, d, f, gradf, c0, c1, t0)
        
        #save old x
        oldx = x
        
        #update values
        x = x + step_size * d
        err = np.linalg.norm(g)
        
        #update B
        s = x - oldx
        y = gradf(x) - grad_f_at_x
        B_y = np.dot(B,y)
        B = B - np.outer(B_y,B_y)/(np.dot(y,B_y)) + np.outer(s,s)/np.dot(y,s)

        #record history
        err_his.append(err)
        func_his.append(fval)
        curr_time = time.time()
        time_his.append(curr_time - start)
        

    return x, func_his, time_his, err_his

def BFGS(x0,f,gradf,c0,c1,t0, TOL):
    """BFGS with weak Wolfe line bisection search

    Parameters
    ----------
    x0 : initialization
    f : objective function
    gradf : gradient of f
    c0 : Parameter for W1 condition (you may want to set a default)
    c1 : Parameter for W2 condition (you may want to set a default)
    t0 : Initial stepsize for line search (you may want to set a default)
    """
    tol = TOL
    x = x0
    err = np.linalg.norm(gradf(x))
    err_his = []
    func_his = []
    time_his = []
    IDENT = np.eye(len(x))
    B = IDENT

    #record history
    err_his.append(err)
    func_his.append(f(x))

    start = time.time()
    while err >= tol:
        #set up
        grad_f_at_x = gradf(x)
        d = -np.dot(B, grad_f_at_x)
        step_size, fval, g = bisection_linesearch(x, d, f, gradf, c0, c1, t0)
        
        #save old x
        oldx = x
        
        #update values
        x = x + step_size * d
        err = np.linalg.norm(g)
        
        #update B
        s = x - oldx
        y = gradf(x) - grad_f_at_x
        B = np.dot(np.dot((IDENT - np.outer(s,y)/np.dot(y,s)), B), (IDENT - np.outer(y,s)/np.dot(y,s))) + np.outer(s,s)/np.dot(y,s)

        #record history
        err_his.append(err)
        func_his.append(fval)
        curr_time = time.time()
        time_his.append(curr_time - start)
        

    return x, func_his, time_his, err_his


def newton(x0,f,gradf,hessf,c0,c1,t0, TOL):
    """Newton's method with weak Wolfe line bisection search

    Parameters
    ----------
    x0 : initialization
    f : objective function
    gradf : gradient of f
    hessf : hessian of f
    c0 : Parameter for W1 condition (you may want to set a default)
    c1 : Parameter for W2 condition (you may want to set a default)
    t0 : Initial stepsize for line search (you may want to set a default)
    """
    tol = TOL
    x = x0
    err = np.linalg.norm(gradf(x))
    err_his = []
    func_his = []
    time_his = []
    
    #record history
    err_his.append(err)
    func_his.append(f(x))

    start = time.time()
    while err >= tol:
        #set up
        grad_f_at_x = gradf(x)
        hess = hessf(x)
        B = np.linalg.inv(hess)
        if np.all(np.linalg.eigvals(hess) > 0) :
            d = -np.dot(B, grad_f_at_x)
        else:
            d = -grad_f_at_x
        
        step_size, fval, g = bisection_linesearch(x, d, f, gradf, c0, c1, t0)
        
        #update values
        x = x + step_size * d
        err = np.linalg.norm(g)
        
        #record history
        err_his.append(err)
        func_his.append(fval)
        curr_time = time.time()
        time_his.append(curr_time - start)
        

    return x, func_his, time_his, err_his
