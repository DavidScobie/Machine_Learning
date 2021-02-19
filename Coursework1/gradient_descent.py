#complete this function
#y is a vector of target values
#x is a vector of input values (same length as y)
#beta is list of length 2 containg the beta coefficients
#alpha is the learning rate
#epsilon is the minimum improvement we require to keep on running the optimization
#maxiter limits the maximum number of iterations (in case we don't converge)
import numpy as np
import sys
def gradient_descent(y, x, beta=[0,0], alpha = 1, epsilon=1e-10, maxiter=10000, Lam=0):
    #unless we start with a specific intialization
    #start with a random solution
    if beta[0] == 0 and beta[1] == 0:
        #use random intiaialization
        beta = np.random.rand(2)
    #our new solution
    beta_new = [0,0]
    #improvement compared to the previous round
    improve = epsilon + 1
    #counter for the iterations
    cnt=0
    #make predictions using the current model
    yhat = beta[0] + x * beta[1]
    #compute the difference between prediction and observation
    error = yhat - y
    #Value of the current cost function [here it is just the RSS]
    ### COMPLETE THIS LINE ###
    # J_current = (1/(2*len(x)))*(np.sum((error)**2))
    J_current = (1/(2*len(x)))*((np.sum((error)**2))+(((beta[1])**2)*Lam))
    #run iterations un
    while (maxiter < 0 or cnt < maxiter) and (improve > epsilon):
        #update rule for the betas
        ### COMPLETE THE UPDATE RULE ###
        beta_new[0] = beta[0] - alpha*((1/(len(x)))*np.sum(error))
        # beta_new[1] = beta[1] - alpha*((1/(len(x)))*np.sum(error*x))
        beta_new[1] = beta[1] - alpha*((1/(len(x)))*np.sum((error*x)+(beta[1]*Lam)))
        #make sure we copy the new betas
        beta = np.copy(beta_new)
        #compute predictions with new beta values
        yhat = beta[0] + x * beta[1]
        #compute the error
        error = yhat - y
        #compute the cost function
        ### COMPLETE THIS LINE (SAME AS ABOVE for J_current) ###
        # J_new = (1/(2*len(x)))*(np.sum((error)**2))
        J_new = (1/(2*len(x)))*((np.sum((error)**2))+(((beta[1])**2)*Lam))
        #compute the improvement compared to previous round
        improve = J_current - J_new
        #update cost of our current 'fit'
        J_current = J_new
        #increase counter
        cnt += 1
        #if our fit is worse than the one before: our alpha is too large
        #issue a warning and continue with a smaller value of alpha
        if improve < 0:
            #if we have negative improvement, our alpha is too large!
            #let's try with half the step size
            sys.stderr.write("observed negative improvement, lowering alpha!\n")
            improve = epsilon + 1
            alpha /= 2
    print(cnt)
    return (beta)