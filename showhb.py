import dssp
from pymol import cmd as cmd
from pymol import stored as stored
import numpy as np
import torch

def show_hb(target="all",thre=0,scale=0.25):
    objs = cmd.get_object_list(target)
    for selection in objs:
        chains=cmd.get_chains(selection)
        for chain in chains:
            stored.resis = []
            #print(chain)
            cmd.iterate(selection + " and name CA " + " and chain " + chain, "stored.resis.append(resi)")
            resis=np.array(stored.resis)
            #print(resis)
            x = torch.tensor(cmd.get_coords(selection + " and name n+ca+c+o " + " and chain " + chain).reshape([1, -1, 4, 3]))
            #print(x.shape)
            hbmap, hcoord = dssp.get_hbond_map(x)
            count=0
            name = cmd.get_unused_name("dist")
            names=[]
            for i_donner in range(0, hcoord.size(1)):
                pname = cmd.get_unused_name("pseudo")
                names.append(pname)
                cmd.pseudoatom(object=pname, pos=hcoord[0].tolist()[i_donner])
                acceptor_resis = resis[(hbmap[0, i_donner + 1, :] > thre) == True]
                #hbmap_i = hbmap[0,i_donner+1,(hbmap[0, i_donner + 1, :] > thre) == True]
                i=0
                for i_acceptor in acceptor_resis:
                    #prob=hbmap_i[i]
                    resinow = i_acceptor
                    cmd.distance(name,
                                 pname,
                                 selection + " and resi " + resinow + " and name o" + " and chain " + chain
                                 )
    #            cmd.set("dash_radius", str(scale*prob.item()), "dist"+str(count).zfill(5))
                    i += 1
                    count += 1
            for pname in names:
                cmd.delete(pname)

pymol.cmd.extend("show_hb", show_hb)
cmd.auto_arg[0]['show_hb'] = cmd.auto_arg[0]['delete']
