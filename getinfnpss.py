import os
import sys
import random
import json
import math
from os.path import join
from math import log
  
########################################
#
#  return {date:[[subgraph], F-socre]}
#
#######################################
  
def getinfnpss(pvalue, network):
    E = network
    subgraph = {}

    for eventdate, place in pvalue.items():
        alpha_max = 0.15
        K = 5
        V = []
        for site, pv in place.items():
            V.append([site, pv])
        V = [V]
    
        C = len(V)
        total_entity = 0
        for lt in V:
            total_entity = total_entity + len(lt)
        Z = int(log(total_entity+1))
        S_STAR = []
        epislon = 0.000001
    
        def quicksort(L):
            if len(L) > 1:
                pivot = random.randrange(len(L))
                elements = L[:pivot]+L[pivot+1:]
                left = [element for element in elements if element[1] < L[pivot][1]]
                right = [element for element in elements if element[1] >= L[pivot][1]]
                return quicksort(left) + [L[pivot]] + quicksort(right)
            return L
    
        for i in range(C):
            V[i] = quicksort(V[i])
    
        for k in range(K):
            for c in range(C):
                if len(V[c]) < k + 1:
                    continue
    
                S = []
                v0 = V[c][k][0]
                S.append(v0)
                S_phi = log(1/(V[c][k][1] + epislon) + epislon)
    
                for z in range(Z):
                    G = []
                    for te in V:
                        for v in te:
                            if v[0] not in S:
                                for e in S:
                                    if E.has_key(str(v[0]) + '_' + str(e)) or E.has_key(str(e) + '_' + str(v[0])):
                                        G.append(v)
                    G = quicksort(G)
                    phi = []
                    for i in range(len(G)):
                        N = i + 1
                        max_phi = -1.0
    
                        for j in range(N):
                            if G[j][1] < alpha_max:
                                a = (j+1)*1.0/N
                                b = G[j][1]
                                s_phi = N * (a * log(epislon + a/(b+epislon)) + (1-a)*log(epislon + (1-a)/(1-b+epislon)))
                                if s_phi > max_phi:
                                    max_phi = s_phi
                        phi.append(max_phi)
    
                    if len(phi) > 0 and max(phi) > S_phi:
                        S_phi = max(phi)
    
                    B = [] + S
                    if len(phi) > 0 and max(phi) > 0:
                        max_phi = max(phi)
                        for i in range(phi.index(max_phi)+1):
                            if G[i][0] not in B:
                                B.append(G[i][0])
    
                    if len(B) - len(S) != 0:
                        S = B
                    else:
                        break
    
                item = []
                item.append(S)
                item.append(S_phi)
                S_STAR.append(item)
    
        S_STAR = quicksort(S_STAR)
        #print S_STAR
        if len(S_STAR) > 0:
            subgraph[eventdate] = S_STAR[len(S_STAR)-1]
    
    return subgraph
