import numpy as np

class myautoencoder():
    '''a simple version of autoencoder
    X--------inputmatrix
    Y--------output matrix
    indim----node number of layer  input
    hdim-----node number of layer hidden
    Z1 = X*W1+B1
    A1 = sigmod(Z1)
    Z2 = A1*W2+B2
    A2 = sigmod(Z2)
    
    '''
    def __init__(self,X,Y,indim,hdim):        
        #training samples
        self.X=X   #input X [[sample1],[sample2],[sample3]]
        self.Y=Y
        #number of samples
        self.M=len(self.X)
        self.indim=indim
        self.hdim=hdim
       
       # initialize the weights randomly and the biases to 0
        r = np.sqrt(6.) / np.sqrt((self.indim * 2) + 1.)
        self.W1 = np.random.random_sample((self.indim,self.hdim)) * 2 * r - r
        self.W2 = np.random.random_sample((self.hdim,self.indim)) * 2 * r - r        
        self.B1 = np.zeros(self.hdim)
        self.B2 = np.zeros(self.indim)
        self.A1 = np.zeros((self.indim,self.hdim))
        self.Z1 = np.zeros((self.indim,self.hdim))        
        self.dW1 = np.zeros((self.indim,self.hdim))
        self.dB1 = np.zeros((self.hdim))
        self.dB2 = np.zeros((self.indim))
        self.dW2 = np.zeros((self.hdim,self.indim))        
        self.A2 = np.zeros((self.indim,self.hdim))
        self.Z2 = np.zeros((self.indim,self.hdim))        
        self.delta2 = np.zeros((self.indim,1))
        self.delta1 = np.zeros((self.hdim,1))
        
        # value of cost function
        self.Jw = 0.0
        # learning rate
        self.alpha = 1.2
        # steps of iteration
        self.steps = 30000
 
    def _sigmoid(self,la):
        '''
        compute the sigmoid function for an array
        of arbitrary shape and size
        '''
        return 1. / (1. + np.exp(-la))
   
    def backpropalgrithom(self):
        #clear values
        self.Jw -= self.Jw
        self.dB1 -= self.dB1
        self.dB2 -= self.dB2
        self.dW1 -= self.dW1
        self.dW2 -= self.dW2
        #backpropagation(iteration over M samples)
        
        # pre-sigmoid activation at the hidden layer
        self.Z1= np.dot(self.X,self.W1) + np.tile(self.B1,(self.M,1))
        # sigmoid activation of the hidden layer
        self.A1 = self._sigmoid(self.Z1)
        # pre-sigmoid activation of the output layer
        self.Z2 = np.dot(self.A1,self.W2) + np.tile(self.B2,(self.M,1))
        # sigmoid activation of the hidden layer
        self.A2 = self._sigmoid(self.Z2)
        #back propagation
        self.delta2 = -(self.X - self.A2) * (self.A2 * (1-self.A2))
        self.Jw += ((self.Y - self.A2)*(self.Y -self.A2)).sum()/self.M
        self.delta1 = np.dot(self.W2,self.delta2.T).T* (self.A1 * (1-self.A1))
        self.dW2 += (self.A1[:,:,np.newaxis]*self.delta2[:,np.newaxis,:]).sum(0)
        self.dW1 += (self.X[:,:,np.newaxis]*self.delta1[:,np.newaxis,:]).sum(0)
        self.dB1 += self.delta1.sum(0)
        self.dB2 += self.delta2.sum(0)
        #uapdate the weight
        self.W1 -= (self.alpha/self.M)*self.dW1
        self.W2 -= (self.alpha/self.M)*self.dW2
        self.B1 -= (self.alpha/self.M)*self.dB1
        self.B2 -= (self.alpha/self.M)*self.dB2
            
    def plainautoencoder(self,steps=None):
        if None == steps:
            steps =self.steps
        for i in range(steps):
            self.backpropalgrithom()
            if i%100==0:
                print "step:%d" % i, "Jw=%f" % self.Jw
    
    def validateautoencoder(self):
        self.Z1= np.dot(self.X,self.W1) + np.tile(self.B1,(self.M,1))
        # sigmoid activation of the hidden layer
        self.A1 = self._sigmoid(self.Z1)
        # pre-sigmoid activation of the output layer
        self.Z2 = np.dot(self.A1,self.W2) + np.tile(self.B2,(self.M,1))
        # sigmoid activation of the hidden layer
        self.A2 = self._sigmoid(self.Z2)
        
        print 'Input Value:'            
        print self.X
        print "Layer Hidden:"
        print self.A1
        print "Output Layer"
        print self.A2

if __name__ == '__main__':
    #x = np.array([[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]])
    #ae=myautoencoder(x,x,4,2)
    x=np.eye(8)
    ae=myautoencoder(x,x,8,3)
    ae.plainautoencoder(10000)
    ae.validateautoencoder()
