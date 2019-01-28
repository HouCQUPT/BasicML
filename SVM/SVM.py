import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KernelFunction:
    """
    Polynomial kernel function: 多项式核函数
    Gaussian kernel function: 高斯核函数
    """
    def __init__(self, x=None, z=None):
        """
        :param x:   a vector of n dimension
                    type: np.ndarray
        :param z:   a vector of n dimension
                    type: np.ndarray
        """
        if np.shape(x) != np.shape(z):
            print("TypeError")
            exit(-1)
        else:
            self.x = x
            self.z = z

    def polynomial(self, p=2):
        """
        Polynomial Kernel Function
        :param p:   int
        :return:    np.array
        """

        # detect type of parameter p
        if not isinstance(p, int):
            print("TypeError\nInput Integer as argument of polynomial kernel function")
            exit(-1)

        x = self.x
        z = self.z
        inner_product = np.matmul(x, z.T)
        kernel_value = np.power(inner_product + 1, p)
        return kernel_value

    def gaussian(self, sigma=2):
        """
        :param sigma: float
        :return:
        """
        x = self.x
        z = self.z
        norm_value = np.linalg.norm(x-z, ord=2)
        kernel_value = np.exp(((-1) * norm_value * norm_value)/(2 * sigma * sigma))
        return kernel_value


class SVMstruct:

    def __init__(self, data, label, c=2, tolerance=0.02, kernel_model='gaussian'):
        self.data = data
        self.label = label
        # m is number of train instance, n is number of dimension of feature vector
        m, n = np.shape(data)
        self.m = m
        self.n = n
        # declare bias
        self.b = 0
        # declare kernel function
        self.kernel = kernel_model
        self.k_value = 0     # kernel function value

        # init Lagrange Multiplier
        self.alpha = np.random.randn(m, 1)

        # init error cache
        self.ecache = np.zeros([m, 1])

        # SMO argument
        self.c = c
        self.tolerance = tolerance


class SVM(SVMstruct):
    def kfunc(self, x, z, kernelobj=KernelFunction):
        """
        the purpose of this function is calculating kernel function value
        :param x:   argument of kernel function, vector
        :param z:   argument of kernel function, vector
        :param kernelobj: Kernel Function Object
        :return:    value
        """
        __arg = 2
        if self.kernel == 'gaussian':
            kernelfunction = kernelobj(x, z)
            self.k_value = kernelfunction.gaussian(sigma=__arg)
            return self.k_value
        elif self.kernel == 'polynomial':
            kernelfunction = kernelobj(x, z)
            self.k_value = kernelfunction.polynomial(p=__arg)
            return self.k_value
        else:
            print("ERROR\nNone Kernel Function:{0}".format(self.kernel))
            exit(-1)

    def gfun(self, x):
        """
        purpose of this function is to calculate value of g(x), 统计学习方法 P127 7.104
        :param x: vector
        :return:
        """
        m = self.m
        _sum = 0
        for i in range(m):
            _sum += self.alpha[i] * self.label[i] + self.kfunc(self.data[i, :], x)
        return _sum + self.b

    def __error(self):
        """
        the purpose is to calculate error cache.
        :return: None
        """
        m = self.m
        for i in range(m):
            self.ecache[i] = self.gfun(self.data[i, :]) - self.label[i]

    def update_error(self, index_i):
        """
        purpose is to update ecache[i]
        :param index_i:
        :return:
        """
        m = self.m
        for i in range(m):
            if i == index_i:
                self.ecache[i] = self.gfun(self.data[i, :]) - self.label[i]
            else:
                pass

    def op_error(self):
        self.__error()
        print(self.ecache[10])

    def low_high(self, index_i, index_j):
        """
        the purpose is to calculate L and H.
        :param index_i:
        :param index_j:
        :return:
        """
        alpha_i = self.alpha[index_i]
        alpha_j = self.alpha[index_j]
        y_i = self.label[index_i]
        y_j = self.label[index_j]
        c = self.c
        if y_i == y_j:
            L = max(0, alpha_j + alpha_i - c)
            H = max(c, alpha_i + alpha_j)
        else:
            L = max(0, alpha_j - alpha_i)
            H = max(c, c + alpha_j - alpha_i)
        return L, H

    def takestep(self, index_i, index_j):
        """
        the purpose is to optimize Lagrange Multiplier
        :return:
        """
        e1 = self.ecache[index_i]
        e2 = self.ecache[index_j]
        alpha_i = self.alpha[index_i]
        alpha_j = self.alpha[index_j]
        y_i = self.label[index_i]
        y_j = self.label[index_j]
        eta = self.kfunc(alpha_i, alpha_i) + self.kfunc(alpha_j, alpha_j) - 2 * self.kfunc(index_i, index_j)
        alpha_new_j = 0
        if eta > 0:
            alpha_j_unc = alpha_j + (y_j * (e1 - e2)) / eta
            L, H = self.low_high(index_i, index_j)
            if alpha_j_unc > H:
                alpha_new_j = alpha_j_unc
            elif alpha_j_unc < L:
                alpha_new_j = L
            else:
                alpha_new_j = alpha_j_unc
        elif eta < 0:
            pass
        else:
            pass

        # update alpha_1
        alpha_i = alpha_i + y_i * y_j * (alpha_j - alpha_new_j)
        alpha_j = alpha_new_j
        self.alpha[index_i], self.alpha[index_i] = alpha_i, alpha_j

    def outer_loop(self, index_i):
        """
        purpose is to chose other Lagrange Multiplier alpha_j. non heuristic method!
        :param index_i:
        :return: alpha_j
        """
        m = self.m
        y_i = self.label[index_i]
        a_i = self.alpha[index_i]
        e1 = self.ecache[index_i]
        r1 = e1 * y_i
        maxabs = -1
        index_j = -1
        # |e1 - e2|
        for i in range(m):
            if i != index_i:
                temp_sub = np.abs(e1 - self.ecache[i])
                if maxabs < temp_sub:
                    index_j = i
        return index_j

    def train_once(self):
        """
        purpose is to run SMO
        :return:
        """
        m = self.m
        b_i_new = -1
        b_j_new = -1
        for i in range(m):
            index_i = i
            index_j = self.outer_loop(index_i)
            if index_j != -1:
                a1_old = self.alpha[index_i]
                a2_old = self.alpha[index_j]
                self.takestep(index_i, index_j)     # update Lagrange Multi
                a1_new = self.alpha[index_i]
                a2_new = self.alpha[index_j]
                if (0 < a1_new < self.c) and (0 < a2_new <self.c):
                    b1_new, b2_new = self.update_b(index_i, index_j, a1_old, a2_old)
                    self.b = b1_new             # update b
                else:
                    b1_new, b2_new = self.update_b(index_i, index_j, a1_old, a2_old)
                    self.b = (b1_new + b2_new) / 2  #update b
                # update ecache
                self.update_error(index_i)
                self.update_error(index_j)

    def update_b(self, index_i, index_j, a1_old, a2_old):
        """
        更新阈值b
        :param index_i:
        :param index_j:
        :return:
        """
        e1, e2 = self.ecache[index_i], self.ecache[index_j]
        y1, y2 = self.label[index_i], self.label[index_j]
        a1, a2 = self.alpha[index_i], self.alpha[index_j]
        data1, data2 = self.data[index_i], self.data[index_j]
        k11, k21 = self.kfunc(data1, data1), self.kfunc(data2, data1)
        k12, k22 = self.kfunc(data1, data2), self.kfunc(data2, data2)
        b1_new = -1 * e1 - y1 * k11 * (a1 - a1_old) - y2*k21(a2 - a2_old) + self.b          # 统计学习方法 P130 7.115
        b2_new = -1 * e2 - y1 * k12 * (a1 - a1_old) - y2*k22(a2 - a2_old) + self.b          # 统计学习方法 P130 7.116
        return b1_new, b2_new




