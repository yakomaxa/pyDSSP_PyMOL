import pydssp as dssp
from pymol import cmd as cmd
from pymol import stored as stored
import numpy as np
import torch

def dsspy(selection="polymer.protein"):
    #cmd.delete("all")
    #cmd.load("tmp.pdb")
    x=torch.tensor(cmd.get_coords(selection + " and name n+ca+c+o").reshape([1,-1, 4, 3]))
    result=dssp.assign(x,"onehot")[0]
    #print(result)
    stored.resis=[]
    cmd.iterate(selection + " and name CA", "stored.resis.append(resi)")
    for i in range(0,len(result)):
        if (all(result[i] == [True, False, False])):
            ss="L"
        elif (all(result[i] == [False, True, False])):
            ss="H"
        elif (all(result[i] == [False, False, True])):
            ss="S"
        else:
            print("illegal assignment")
        resinow=stored.resis[i]
        cmd.alter(selection + " and resi "+resinow,"ss="+"'"+ss+"'")
    cmd.rebuild(selection)
pymol.cmd.extend("dsspy", dsspy)
cmd.auto_arg[0]['dsspy'] = cmd.auto_arg[0]['delete']
