# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:52:31 2021

@author: jelmer
"""

import numpy as np

class KDTree:
    class Node:
        def __init__(self, pos, ids):
            if len(pos) > 20:
                # find splitting axis
                var = np.var(pos,0)
                means = np.mean(pos,0)
                best_axis = np.argmax(var)
                
                which = pos[:,best_axis] > means[best_axis]
                self.pos = None
                self.axis = best_axis
                self.divider = means[best_axis]
                self.nodes = [
                    type(self)(pos[which], ids[which]),
                    type(self)(pos[np.logical_not(which)], ids[np.logical_not(which)])
                ]
            else:
                self.pos = pos
                self.ids = ids
        
        def collect(self):
            if self.pos is not None:
                return [self.pos]
            else:
                r = []
                for n in self.nodes:
                    r.extend(n.collect())
                return r
                
        def print(self, depth=0):
            
            if self.pos is not None:
                print(' '*depth + f'leaf with {len(self.pos)} positions')
            else:
                print(' '*depth + f'divider={self.divider:.1f} axis={self.axis}')
                for n in self.nodes:
                    n.print(depth+1)
                    
                
        def selectInRadius(self, center, radius):
            if self.pos is not None:
                return self.ids[ np.nonzero( ((self.pos-center)**2).sum(1) <= radius**2 )[0] ]
            
            if center[self.axis] + radius > self.divider:
                a = self.nodes[0].selectInRadius(center, radius)
            else:
                a = np.zeros(0, dtype=np.int32)

            if center[self.axis] - radius <= self.divider:
                b = self.nodes[1].selectInRadius(center, radius)
            else:
                b = np.zeros(0, dtype=np.int32)

            return np.concatenate((a,b))
        
    def __init__(self, positions):
        self.pos = positions
        self.root = self.Node(positions, np.arange(len(positions)))
    
    def selectInRadius(self, center, radius):
        center = np.array(center)
        idx = self.root.selectInRadius(center, radius)
        return self.pos[idx], idx


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    np.random.seed(0)
    
    xy = np.random.uniform([0,0], [20,20], size=(2000,2))
    #plt.scatter(xy[:,0],xy[:,1], c='b')

    tree = KDTree(xy)
    
    r = tree.root.collect()
    col = np.concatenate( [np.ones(len(x))*i for i,x in enumerate(r)] )

    pos = np.concatenate(r)
    plt.scatter(pos[:,0], pos[:,1], c=col)
    
    pos,idx = tree.selectInRadius([10,10], 5)
    plt.scatter(pos[:,0], pos[:,1], c='red')
    