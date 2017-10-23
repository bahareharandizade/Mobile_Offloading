import numpy as np
import random
from collections import defaultdict
from scipy.stats import beta
from matplotlib import pyplot as plt


nodes=dict()
n = 5
iteration = 1000
rand = np.random.uniform(0,1,n)# probability of acceptance
pthreshold = 0.2
#*************************************************************************

def weightedAvg1(A,B,index=-1):
    aaa = []
    bbb = []
    if index == -1:# the final alpha and beta
        alpha = set(nodes[A]['interactions'][B]['alpha'])
        Beta = set(nodes[A]['interactions'][B]['Beta'])
        for item in alpha:
            aaa.append(item)
        for item in Beta:
            bbb.append(item)
        alphaT = nodes[A]['interactions'][B]['alphaT']
        BetaT = nodes[A]['interactions'][B]['BetaT']



    else: #alpha and beta in each T
        alpha = nodes[A]['interactions'][B]['alpha'][:index+1]
        Beta = nodes[A]['interactions'][B]['Beta'][:index+1]
        alpha = set(alpha)
        Beta = set(Beta)
        alphaT = nodes[A]['interactions'][B]['alphaT'][:len(alpha)]
        BetaT = nodes[A]['interactions'][B]['BetaT'][:len(Beta)]
        for item in alpha:
            aaa.append(item)
        for item in Beta:
            bbb.append(item)

    return np.average(aaa, weights=alphaT), np.average(bbb, weights=BetaT)

#*********************************************************************************


def weightedAvg2(A,B,index = -1):
    aaa = []
    bbb = []
    if index == -1:

        alpha = set(nodes[A]['interactions'][B]['alpha1'])
        Beta = set(nodes[A]['interactions'][B]['Beta1'])
        for item in alpha:
            aaa.append(item)
        for item in Beta:
            bbb.append(item)



    else:
        alpha = nodes[A]['interactions'][B]['alpha1'][:index + 1]
        Beta = nodes[A]['interactions'][B]['Beta1'][:index + 1]
        alpha = set(alpha)
        Beta = set(Beta)
        for item in alpha:
            aaa.append(item)
        for item in Beta:
            bbb.append(item)


    return np.average(aaa, weights=aaa),np.average(bbb, weights=bbb)

#**********************************************************************************

def expected(a,b):
    return a/(a+b)

#**********************************************************************************

def Variance(a,b):
    x = a*b
    y = x**2
    z = a+b+1
    return x/(y*z)

#**********************************************************************************

def plotstimation(mode):
    # plot stimation

    for item in nodes:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        for ax, intnode in zip(axes.ravel(), nodes[item]['interactions'].keys()):
            if mode == 1:
                a, b = weightedAvg1(item, intnode)
            else:
                a, b = weightedAvg2(item, intnode)
            x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
            ax.margins(0.1)
            ax.plot(x, beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='beta pdf')
            ax.axvline(nodes[intnode]['accprob'], color='k', linestyle='--')
            ax.set_xlabel('x')
            ax.set_title(
                'node ' + str(intnode) + ' probability: ' + str(round(nodes[intnode]['accprob'], 2)) + ' ,alpha:' + str(round(a, 2)) + ' ,beta:' + str(round(b, 2)), fontsize=10)
            ax.set_xlim((0.0, 0.99))
            ax.set_xticks(np.arange(0.0, 0.99, 0.1))

        plt.tight_layout()
        plt.savefig('mode: '+str(mode)+'_'+str(item) +'estimation' +'.png')
        plt.clf()

# **********************************************************************************
def ploterror(mode):

    # plot Error

    for item in nodes:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        for ax, intnode in zip(axes.ravel(), nodes[item]['interactions'].keys()):
            E = []
            real = nodes[intnode]['accprob']
            if mode ==1:
                for i in range(0, len(nodes[item]['interactions'][intnode]['alpha'])):
                    al, be = weightedAvg1(item, intnode, i)
                    E.append(expected(al, be) - real)
            else:
                for i in range(0, len(nodes[item]['interactions'][intnode]['alpha1'])):
                    al, be = weightedAvg2(item, intnode, i)
                    E.append(expected(al, be) - real)


            aa = []
            bb = []
            if mode ==1:

                alphat = nodes[item]['interactions'][intnode]['alphaT']
                betat = nodes[item]['interactions'][intnode]['BetaT']
                for h in alphat:
                    aa.append(h)
                for m in betat:
                    bb.append(m)
                L = aa + bb
                time = sorted(set(L))
            else:
                time = range(0, len(nodes[item]['interactions'][intnode]['alpha1']))

            x = time
            ax.margins(0.5)
            ax.plot(x, E, 'ro')
            ax.set_xlabel('Time')
            ax.set_title('node ' + str(intnode) + ' probability: ' + str(round(nodes[intnode]['accprob'], 2)),
                         fontsize=10)
            if mode == 1:

                ax.set_xlim((0, 1001))
                ax.set_xticks(np.arange(0, 1001, 100))
            else:
                ax.set_xlim((0, 71))
                ax.set_xticks(np.arange(0, 71, 5))

        plt.tight_layout()
        plt.savefig('mode: '+str(mode)+'_'+str(item) + 'error' + '.png')
        plt.clf()

# **********************************************************************************

def plotVariance(mode):
    # plot Variance
    for item in nodes:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        for ax, intnode in zip(axes.ravel(), nodes[item]['interactions'].keys()):
            V = []

            if mode == 1:
                for i in range(0, len(nodes[item]['interactions'][intnode]['alpha'])):
                    al, be = weightedAvg1(item, intnode, i)
                    V.append(Variance(al,be))
            else:
                for i in range(0, len(nodes[item]['interactions'][intnode]['alpha1'])):
                    al, be = weightedAvg2(item, intnode, i)
                    V.append(Variance(al,be))

            aa = []
            bb = []
            if mode == 1:

                alphat = nodes[item]['interactions'][intnode]['alphaT']
                betat = nodes[item]['interactions'][intnode]['BetaT']
                for h in alphat:
                    aa.append(h)
                for m in betat:
                    bb.append(m)
                L = aa + bb
                time = sorted(set(L))
            else:
                time = range(0, len(nodes[item]['interactions'][intnode]['alpha1']))

            x = time

            ax.plot(x, V, 'ro')
            ax.margins(0.05)
            ax.set_xlabel('Time')
            ax.set_title('node ' + str(intnode) + ' probability: ' + str(round(nodes[intnode]['accprob'], 2)),
                         fontsize=10)
            if mode == 1:

                ax.set_xlim((0, 1001))
                ax.set_xticks(np.arange(0, 1001, 100))
            else:
                ax.set_xlim((0, 71))
                ax.set_xticks(np.arange(0, 71, 5))

        plt.tight_layout()
        plt.savefig('mode: '+str(mode)+'_'+str(item) + 'variance' + '.png')
        plt.clf()

# **********************************************************************************
#Initialization

for item in range(0,n): # just for initialization
    item = str(item)
    nodes[item] = dict()
    nodes[item]['accprob'] = rand[int(item)]
    nodes[item]['interactions'] = dict()

for A in range(0,n):
    for B in range(0,n):
        A = str(A)
        B = str(B)
        if A != B:

            nodes[A]['interactions'][B] = defaultdict(list)
            nodes[A]['interactions'][B]['alpha'].append(1)
            nodes[A]['interactions'][B]['Beta'].append(1)
            nodes[A]['interactions'][B]['alphaT'].append(1)
            nodes[A]['interactions'][B]['BetaT'].append(1)
            nodes[A]['interactions'][B]['alpha1'].append(1)
            nodes[A]['interactions'][B]['Beta1'].append(1)

#**************************************************************************************
#Iteration starts


for T in range(2,iteration):
    randnodes = random.sample(range(0, n), 2)
    A = str(randnodes[0]) # try to affload
    B = str(randnodes[1]) # get or reject!

    if B in nodes[A]['interactions']:
        a,b = weightedAvg1(A,B)
        alpha = a
        Beta = b


    x = np.linspace(beta.ppf(0.01, alpha, Beta), beta.ppf(0.99, alpha, Beta), 100)# random variable netween 0 1 --> beta is arranged to estimate probability of acceptance of node B
    cdf = beta.cdf(x, alpha, Beta)
    mthreshod = [j for j in x if j>=0.4]#this should be set!

    index = len(x) - len(mthreshod) # the first number in x more than threshold
    prob=1- cdf[index]# what is the prob(X>p)

    if prob >= pthreshold:
        # pass the task to B
        guessB = np.random.uniform(0, 1, 1)
        if guessB <= nodes[B]['accprob']:# your guess is true!
            index=len(nodes[A]['interactions'][B]['alpha'])-1
            nodes[A]['interactions'][B]['alpha'].append(nodes[A]['interactions'][B]['alpha'][index]+1)#increase alpha +1
            nodes[A]['interactions'][B]['alphaT'].append(T)#increase T
            if len(nodes[A]['interactions'][B]['Beta']) < len(nodes[A]['interactions'][B]['alpha']):

                nodes[A]['interactions'][B]['Beta'].append(nodes[A]['interactions'][B]['Beta'][index])#for easier calculation of beta in avg weighting
        else: #if we guess wrong
            index = len(nodes[A]['interactions'][B]['Beta']) - 1
            nodes[A]['interactions'][B]['Beta'].append(nodes[A]['interactions'][B]['Beta'][index] + 1)  # increase beta +1
            nodes[A]['interactions'][B]['BetaT'].append(T)  # increase T
            if len(nodes[A]['interactions'][B]['alpha']) < len(nodes[A]['interactions'][B]['Beta']):
                nodes[A]['interactions'][B]['alpha'].append(nodes[A]['interactions'][B]['alpha'][index])



    if B in nodes[A]['interactions']:
            a, b = weightedAvg2(A, B)
            alpha = a
            Beta = b
    x = np.linspace(beta.ppf(0.01, alpha, Beta), beta.ppf(0.99, alpha, Beta),100)  # random variable netween 0 1 --> beta is arranged to estimate probability of acceptance of node B
    cdf = beta.cdf(x, alpha, Beta)
    mthreshod = [j for j in x if j >= 0.4]  # this should be set!

    index = len(x) - len(mthreshod)  # the first number in x more than threshold
    prob = 1 - cdf[index]  # what is the prob(X>p)

    if prob >= pthreshold:
        # pass the task to B
        if guessB <= nodes[B]['accprob']:  # your guess is true!

            index = len(nodes[A]['interactions'][B]['alpha1']) - 1
            nodes[A]['interactions'][B]['alpha1'].append(nodes[A]['interactions'][B]['alpha1'][index] + 1)  # increase alpha +1
            if len(nodes[A]['interactions'][B]['Beta1']) < len(nodes[A]['interactions'][B]['alpha1']):
                nodes[A]['interactions'][B]['Beta1'].append(nodes[A]['interactions'][B]['Beta1'][index])  # for easier calculation of beta in avg weighting
        else:  # if we guess wrong

            index = len(nodes[A]['interactions'][B]['Beta1']) - 1
            nodes[A]['interactions'][B]['Beta1'].append(nodes[A]['interactions'][B]['Beta1'][index] + 1)  # increase beta +1
            if len(nodes[A]['interactions'][B]['alpha1']) < len(nodes[A]['interactions'][B]['Beta1']):
                nodes[A]['interactions'][B]['alpha1'].append(nodes[A]['interactions'][B]['alpha1'][index])
#**********************************************************
#print output

for item in nodes:
   print 'id node:'+str(item)
   # print 'probability:'+str(nodes[item]['accprob'])
   # print 'length:' + str(len(nodes[item]['interactions']))
   for node,list in nodes[item]['interactions'].iteritems():

       print 'id: '+str(node) + ' prob: '+str(nodes[node]['accprob'])+' alpha: '+str(list['alpha'][len(list['alpha'])-1])+' Beta: '+str(list['Beta'][len(list['Beta'])-1])

#**********************************************************


plotstimation(1)
plotstimation(2)
ploterror(1)
ploterror(2)
plotVariance(1)
plotVariance(2)





#**********************************************************
#**********************************************************
