import numpy as np
import math



class TwoLayerNet():
    """
    2 Layer Network를 만드려고 합니다.

    해당 네트워크는 아래의 구조를 따릅니다.

    input - Linear - ReLU - Linear - Softmax

    Softmax 결과는 입력 N개의 데이터에 대해 개별 클래스에 대한 확률입니다.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        네트워크에 필요한 가중치들을 initialization합니다.
        initialized by random values
        해당 가중치들은 self.params 라는 Dictionary에 담아둡니다.

        input_size: 데이터의 변수 개수 - D
        hidden_size: 히든 층의 H 개수 - H
        output_size: 클래스 개수 - C

        """
        # A = X*W1+b1 (N*hidden_size)
        # B = ReLU(A) (N*hidden_size)
        # C = B*W2+b1 (N*output_size)
        # D = softmax(C) (N*output_size)
        
        self.params = {}
        self.params["W1"] = std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.random.randn(hidden_size)
        self.params["W2"] = std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.random.randn(output_size)

    
    def forward(self, X, y=None):
        """
        X: input 데이터 (N, D)
        y: 레이블 (N,)
    
        Linear - ReLU - Linear - Softmax - CrossEntropy Loss
    
        y가 주어지지 않으면 Softmax 결과 p와 Activation 결과 a를 return합니다. p와 a 모두 backward에서 미분할때 사용합니다.
        y가 주어지면 CrossEntropy Error를 return합니다.
    
        """
    
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        N, D = X.shape
    
        # 여기에 p를 구하는 작업을 수행하세요.
    
        h = np.dot(X, W1) + b1  # linear: h = X*W1+b1 (N*hidden_size)
        a = np.maximum(h, 0)  # relu:   a = ReLU(h) (N*hidden_size)
        o = np.dot(a, W2) + b2  # linear: o = a*W2+b2 (N*output_size)
        p = np.exp(o) / (np.exp(o).sum(axis=1).reshape(-1, 1))
    
        # y가 주어지지 않으면 p와 a를 return합니다.
        if y is None:
            return p, a
    
        # 여기에 Loss(cross entropy)를 구하는 작업을 수행하세요.
    
        onehot_y = np.eye(p.shape[1])[y]  # y를 one_hot encoding
        Loss = (-np.log(p) * onehot_y).sum(axis=1).mean()
    
        # print('loss : ',Loss)
    
        return Loss
    
    
    def backward(self, X, y, learning_rate=1e-5):
        """
    
        X: input 데이터 (N, D)
        y: 레이블 (N,)
    
        grads에는 Loss에 대한 W1, b1, W2, b2 미분 값이 기록됩니다.
    
        원래 backw 미분 결과를 return 하지만
        여기서는 Gradient Descent방식으로 가중치 갱신까지 합니다.
    
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        N = X.shape[0]  # 데이터 개수
        grads = {}
    
        p, a = self.forward(X)  # y=None
    
        # 여기에 파라미터에 대한 미분을 저장하세요.
    
        dp = p.copy() 
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                if (j == y[i]):
                    dp[i][j] -= 1
    
        #da = np.heaviside(a, 0)
        da = np.dot(dp, W2.T)
    
        grads["W2"] = np.dot(a.T, dp)
        grads["b2"] = np.sum(dp, axis=0)
        grads["W1"] = np.dot(X.T, da)
        grads["b1"] = np.sum(da, axis=0)
    
        # weight update (우리 모델에서 weight는 W1, b1, W2, b2뿐)
        self.params["W2"] -= learning_rate * grads["W2"]
        self.params["b2"] -= learning_rate * grads["b2"]
        self.params["W1"] -= learning_rate * grads["W1"]
        self.params["b1"] -= learning_rate * grads["b1"]
    
    
    def accuracy(self, X, y):
        p, _ = self.forward(X)
        pre_p = np.argmax(p, axis=1)  # 예측된 y값 (softmax의 값이 가장 큰 값으로 예측)
    
        return np.sum(pre_p == y) / pre_p.shape[0]  # accuracy


class ThreeLayerNet():
    
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, std=1e-4):
        # weight initialize
        self.params = {}
        self.params["W1"] = std * np.random.randn(input_size, hidden1_size)
        self.params["b1"] = np.random.randn(hidden1_size)
        self.params["W2"] = std * np.random.randn(hidden1_size, hidden2_size)
        self.params["b2"] = np.random.randn(hidden2_size)
        self.params["W3"] = std * np.random.randn(hidden2_size, output_size)
        self.params["b3"] = np.random.randn(output_size)

    
    def forward(self, X, y=None):
        """
        X: input data, y: target
        Linear - ReLU - Linear - ReLU - Linear - Softmax - CrossEntropy Loss
        """
    
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]
        N, D = X.shape
    
        h1 = np.dot(X, W1) + b1  # linear
        a1 = np.maximum(h1, 0)   # relu
        h2 = np.dot(a1, W2) + b2   # linear
        a2 = np.maximum(h2, 0)   # relu
        o  = np.dot(a2, W3) + b3  # linear: o = a*W2+b2 (N*output_size)
        p  = np.exp(o) / (np.exp(o).sum(axis=1).reshape(-1, 1)) # softmax
    
        if y is None:
            return (p, a1, a2)
        
        onehot_y = np.eye(p.shape[1])[y]  # y를 one_hot encoding
        Loss = (-np.log(p) * onehot_y).sum(axis=1).mean() # cross entropy
        return Loss
    
    
    def backward(self, X, y, learning_rate=1e-5):
        # X : input data, y : target 
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]
        N = X.shape[0]  # num of data
        grads = {} # gradient 
        
        p, a1, a2 = self.forward(X)  # y=None
        
        do = p.copy()
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                if (j == y[i]):
                    do[i][j] -= 1

        da2 = np.dot(do, W3.T)
        da1 = np.dot(da2, W2.T)
        
        # gradient
        grads["W3"] = np.dot(a2.T, do)
        grads["b3"] = np.sum(do, axis=0)
        grads["W2"] = np.dot(a1.T, da2)
        grads["b2"] = np.sum(da2, axis=0)
        grads["W1"] = np.dot(X.T, da1)
        grads["b1"] = np.sum(da1, axis=0)
    
        # weight update (gradient descent)
        self.params["W3"] -= learning_rate * grads["W3"]
        self.params["b3"] -= learning_rate * grads["b3"]
        self.params["W2"] -= learning_rate * grads["W2"]
        self.params["b2"] -= learning_rate * grads["b2"]
        self.params["W1"] -= learning_rate * grads["W1"]
        self.params["b1"] -= learning_rate * grads["b1"]
    
    
    def accuracy(self, X, y):
        p, _, _ = self.forward(X)
        pre_p = np.argmax(p, axis=1)  
        return np.sum(pre_p == y) / pre_p.shape[0]  # accuracy

    

class FourLayerNet():
    
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, output_size, std=1e-4):
        # weight initialize
        self.params = {}
        self.params["W1"] = std * np.random.randn(input_size, hidden1_size)
        self.params["b1"] = np.random.randn(hidden1_size)
        self.params["W2"] = std * np.random.randn(hidden1_size, hidden2_size)
        self.params["b2"] = np.random.randn(hidden2_size)
        self.params["W3"] = std * np.random.randn(hidden2_size, hidden3_size)
        self.params["b3"] = np.random.randn(hidden3_size)
        self.params["W4"] = std * np.random.randn(hidden3_size, output_size)
        self.params["b4"] = np.random.randn(output_size)

    
    def forward(self, X, y=None):

        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]
        W4, b4 = self.params["W4"], self.params["b4"]
        N, D = X.shape
    
        h1 = np.dot(X, W1) + b1  # linear
        a1 = np.maximum(h1, 0)   # relu
        h2 = np.dot(a1, W2) + b2   # linear
        a2 = np.maximum(h2, 0)   # relu
        h3 = np.dot(a2, W3) + b3
        a3 = np.maximum(h3, 0)
        o  = np.dot(a3, W4) + b4  # linear: o = a*W2+b2 (N*output_size)
        p  = np.exp(o) / (np.exp(o).sum(axis=1).reshape(-1, 1)) # softmax
    
        if y is None:
            return (p, a1, a2, a3)
        
        onehot_y = np.eye(p.shape[1])[y]  # y를 one_hot encoding
        Loss = (-np.log(p) * onehot_y).sum(axis=1).mean() # cross entropy
        return Loss
    
    
    def backward(self, X, y, learning_rate=1e-5):
        # X : input data, y : target 
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]
        W4, b4 = self.params["W4"], self.params["b4"]
        N = X.shape[0]  # num of data
        grads = {} # gradient 
        
        p, a1, a2, a3 = self.forward(X)  # y=None
        
        do = p.copy()
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                if (j == y[i]):
                    do[i][j] -= 1

        da3 = np.dot(do, W4.T)
        da2 = np.dot(da3, W3.T)
        da1 = np.dot(da2, W2.T)
        
        # gradient
        grads["W4"] = np.dot(a3.T, do)
        grads["b4"] = np.sum(do, axis=0)
        grads["W3"] = np.dot(a2.T, da3)
        grads["b3"] = np.sum(da3, axis=0)
        grads["W2"] = np.dot(a1.T, da2)
        grads["b2"] = np.sum(da2, axis=0)
        grads["W1"] = np.dot(X.T, da1)
        grads["b1"] = np.sum(da1, axis=0)
    
    
        # weight update (gradient descent)
        self.params["W4"] -= learning_rate * grads["W4"]
        self.params["b4"] -= learning_rate * grads["b4"]
        self.params["W3"] -= learning_rate * grads["W3"]
        self.params["b3"] -= learning_rate * grads["b3"]
        self.params["W2"] -= learning_rate * grads["W2"]
        self.params["b2"] -= learning_rate * grads["b2"]
        self.params["W1"] -= learning_rate * grads["W1"]
        self.params["b1"] -= learning_rate * grads["b1"]
    
    
    def accuracy(self, X, y):
        p, _, _, _ = self.forward(X)
        pre_p = np.argmax(p, axis=1)  
        return np.sum(pre_p == y) / pre_p.shape[0]  # accuracy
