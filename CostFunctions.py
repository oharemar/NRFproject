import numpy as np

def sigmoid(z,gamma): # sigmoid function, gamma is hyperparameter
    return 1.0/(1.0+ np.exp(-gamma*z))
def sigmoid_prime (z,gamma): # derivation of sigmoid function
    """Derivative of the sigmoid function."""
    return gamma*(sigmoid(z,gamma)*(1- sigmoid(z,gamma)))

def sigmoid_inverse(z,gamma):
    x = z.reshape(1, -1).tolist()[0]
    res = np.zeros((len(x), 1), dtype=np.float64).reshape(-1, 1)
    for val, index in zip(x, range(len(x))):
        if val < 10 ** (-5):
            res[index, 0] = -(10)
            continue
        if val > 0.99:
            res[index, 0] = 10
            continue
        else:
            res[index, 0] = (1/gamma)*np.log(val/(1-val))
    return res

def tanh(z,gamma): # tangent hyperbolic function, gamma is hyperparameter
    ret = (np.exp(2.0*gamma*z) - 1.0)/(np.exp(2.0*gamma*z) + 1.0)
    ret = np.where(np.isnan(ret),1,ret)
    return ret
def derivative_tanh(z,gamma): # derivation of tanh function, gamma is hyperparameter
    return gamma*(1.0 - (tanh(z,gamma))**2.0)

def softmax(z,gamma = 1):
    sum = np.sum(np.exp(gamma*z))
    return np.exp(gamma*z)/sum

def softmax_inverse(z,gamma=1):
    x = z.reshape(1,-1).tolist()[0]
    res = np.zeros((len(x),1),dtype=np.float64).reshape(-1,1)
    for val,index in zip(x,range(len(x))):
        if val < 10**(-5):
            res[index,0] = -(10)
            continue
        if val > 0.99:
            res[index,0] = 10
            continue
        else:
            excluded_sum = sum([np.exp(gamma*x[ind]) for ind in range(len(x)) if ind != index])
            arg = excluded_sum * (val/(1-val))
            res[index, 0] = (1/gamma)*np.log(arg)
    return res


def vectorized_result(y,num_classes):
    vec_res = np.zeros((num_classes,1), dtype=float)
    vec_res[int(y),0] = 1
    return vec_res

class CrossEntropyCost():
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ‘‘a‘‘ and desired output ‘‘y‘‘.
        Note that np.nan_to_num is used to ensure numerical stability.
        In particular, if both ‘‘a‘‘ and ‘‘y‘‘ have a 1.0 in the same slot, then the expression
        (1-y)*np.log(1-a) returns nan. The np.nan_to_num ensures that that is converted to the correct value (0.0)."""
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z,a,y,gamma):
        """Return the error delta from the output layer.
        Note that the parameter ‘‘z‘‘ is not used by the method. It is included in the method’s parameters
        in order to make the interface consistent with the delta method for other cost classes.
        """
        return gamma*(a - y)

class QuadraticCost():
    @staticmethod
    def fn(a,y):
        """Return the cost associated with an output ‘‘a‘‘ and desired output ‘‘y‘‘."""
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z,a,y,gamma):
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z,gamma) # this is working only for output layer with sigmoid function

class LogLikelihoodCost():
    @staticmethod
    def fn(a,y):
        """Return the cost associated with an output ‘‘a‘‘ and desired output ‘‘y‘‘."""
        index = np.argmax(y)
        if a[index,0] != 0:
            return -np.log(a[index,0])
        else:
            return 10 ** 8

    @staticmethod
    def delta(z, a, y, gamma):
        """Return the error delta from the output layer."""
        return gamma*(a - y)

