import numpy as np

#Initialization
n
gamma
alpha
W
w0
P

def step(V: np.array[n]) :
    V_offer_next = np.zeros(n)
    for i in range(n) :
        Q_accept = (1+gamma*alpha/(1-gamma*alpha))*W[i] + gamma*[alpa*(1-alpa)*(1+gamma**2)*np.dot(P*V)]
        Q_decline = w0 + gamma*np.dot(P,V)
        V_offer_next = max(Q_accept, Q_decline)
    return(V_offer_next)

def value_iteration_result()

V_0 = np.zeros(n)

V_employed_optimal = 1/(1-gamma*alpha)*(W[i]+gamma*(1-alpha)*np.dot(P, V_offer_optimal))
