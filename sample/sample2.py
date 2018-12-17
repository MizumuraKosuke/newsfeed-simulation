# coding: UTF-8
import math, random
import numpy as np
import matplotlib.pyplot as plt

class User:
    def __init__(self, spin, x, y):
        self.spin = spin
        self.x = x
        self.y = y

class Field:
    def __init__(self, x, y, j):
        self.x = x
        self.y = y
        self.j = j
        self.users = np.zeros((self.x,self.y), dtype=object)

    def add_users(self, spin, x, y):
        self.users[x, y] = User(spin, x, y)

    def four_way(self, user):
        adj = []
        if (user.x - 1) >= 0:
            adj.append(self.users[(user.x)-1, user.y])
        else:
            adj.append(self.users[(self.x)-1, user.y])

        if (user.y - 1) >= 0:
            adj.append(self.users[user.x, (user.y)-1])
        else:
            adj.append(self.users[user.x, (self.y)-1])

        if (user.x + 1) < np.size(self.users,0):
            adj.append(self.users[(user.x)+1, user.y])
        else:
            adj.append(self.users[0, user.y])

        if (user.y + 1) < np.size(self.users,0):
            adj.append(self.users[user.x, (user.y)+1])
        else:
            adj.append(self.users[user.x, 0])

        dE = self.energy(adj, self.j, user)
        T = self.temp(adj)
        P = self.transition_rate(np.absolute(dE), T)
        
        if dE < 0: #判断を変えた場合のエネルギーが小さかったとき
            user.spin = user.spin * (-1)
        else:
            rand = random.random()
            if rand <= P:
                user.spin = user.spin * (-1)

    def energy(self, adj, j, user):
        E = 0 #判断を変えない場合のエネルギー
        Echange = 0 #判断を変えた場合のエネルギー
        for i in adj:
            E = E + (-1)*j*user.spin*i.spin
            Echange = Echange + j*user.spin*i.spin
        dE = Echange - E #判断を変えた場合と変えない場合のエネルギーの差
        return dE

    def temp(self, adj):
        s = 0
        for i in adj:
            s = s + i.spin
        s = np.absolute(s)
        T = float(1)/(1+2*s)
        return T
    def nor_temp(self, adj):
        T = 1
        return T

    def transition_rate(self, dE, T):
        P = float(np.exp((float(-1)*dE/T)))/2
        return P
            
    def step(self):
        rul = [] #random user list
        for i in self.users:
            rul.extend(i)
        random.shuffle(rul)
        for user in rul:
            self.four_way(user)

    def start(self,n_step):
        for i in range(self.x):
            for j in range(self.y):
                self.add_users(-1, i, j)
        for i in range(n_step):
            self.step()
            cond = self.condition()
            self.timeplot(i+1,cond)
        self.plotend()
    
    def condition(self):
        m = 0
        for i in self.users:
            for user in i:
                m = m + user.spin
        M = float(m)/(self.x*self.y)
        return(M)
    
    def timeplot(self,step,cond):
        plt.ion()
        plt.plot(step,cond,'o')
        plt.draw()
        plt.pause(0.000001)
    def plotend(self):
        plt.ioff()
        plt.show()
                

field = Field(30,30,0.047)
field.start(50000)

def relationplot(jst,jen,jf,x,y,step):
    plt.figure()
    j = jst
    plt.ion()
    while j<jen:
        field = Field(x,y,j)
        field.start(step)
        plt.plot(j,np.absolute(field.condition()),'o')
        plt.draw()
        plt.pause(0.00001)
        j = j + jf
        
    plt.ioff()
    plt.show()

#relationplot(0.03,0.06,0.001,30,30,5000)
        


