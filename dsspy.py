import pydssp as dssp
from pymol import cmd as cmd
from pymol import stored as stored
import numpy as np
import torch

def dsspy(target="polymer.protein"):
     objs = cmd.get_object_list(target)
     for selection in objs:
         chains=cmd.get_chains(selection)
         for chain in chains:
             stored.resis = []
             x=cmd.get_coords(selection + " and name n+ca+c+o and chain " + chain).reshape([1,-1, 4, 3])
             result=dssp.assign(x,"onehot")[0]
             stored.resis=[]
             cmd.iterate(selection + " and name CA and chain " + chain , "stored.resis.append(resi)")
             for i in range(0,len(result)):
                 #print(result[i])
                 if (all(result[i] == [True, False, False])):
                     ss="L"
                 elif (all(result[i] == [False, True, False])):
                     ss="H"
                 elif (all(result[i] == [False, False, True])):
                     ss="S"
                 else:
                     print("illegal assignment")
                 resinow=stored.resis[i]
                 cmd.alter(selection + " and resi "+resinow+" and chain "+chain,"ss="+"'"+ss+"'")
             cmd.rebuild(selection + " and chain " + chain)
                        
pymol.cmd.extend("dsspy", dsspy)
cmd.auto_arg[0]['dsspy'] = cmd.auto_arg[0]['delete']
