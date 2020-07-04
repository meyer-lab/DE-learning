'''implement the basic model simulation'''
from model import Model

test1 = Model()
test1.random_params() # Randomly initialize parameters
test1.sim() # Run ODE model
test1.graph() # Results of simulation

