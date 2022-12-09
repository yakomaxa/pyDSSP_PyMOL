from pymol import cmd as cmd
from pymol import stored as stored
import numpy as np
import torch
from einops import repeat

def show_hb(target="all",thre=0,delete_dummy=1):
    objs = cmd.get_object_list(target)
    for selection in objs:
        chains=cmd.get_chains(selection)
        #print(chains)
        stored.resis = []
        stored.chains = []
        for chain in chains:
            cmd.iterate(selection + " and name CA " + " and chain " + chain, "stored.resis.append(resi)")
            cmd.iterate(selection + " and name CA " + " and chain " + chain, "stored.chains.append(chain)")
        resis=np.array(stored.resis)
        chains_cat="+".join(stored.chains)
        chains=np.array(stored.chains)
        #print(chains_cat)
        x = torch.tensor(cmd.get_coords(selection + " and name n+ca+c+o " + " and chain " + chains_cat).reshape([1, -1, 4, 3]))
        hbmap, hcoord = get_hbond_map(x)
        name = cmd.get_unused_name("dist")
        names=[]
        for i_donner in range(0, hcoord.size(1)):
            pname = cmd.get_unused_name("pseudo")
            names.append(pname)
            cmd.pseudoatom(object=pname, pos=hcoord[0].tolist()[i_donner])
            acceptor_resis = resis[(hbmap[0, i_donner + 1, :] > float(thre)) == True]
            acceptor_chains = chains[(hbmap[0, i_donner + 1, :] > float(thre)) == True]
            for i_acceptor,chain_acceptor in zip(acceptor_resis,acceptor_chains):
                resinow = i_acceptor
                chainnow = chain_acceptor
                cmd.distance(name,
                             pname,
                             selection + " and resi " + resinow + " and name o" + " and chain " + chain_acceptor
                        )
            if (int(delete_dummy)==1):
                for pname in names:
                    cmd.delete(pname)

pymol.cmd.extend("hb", show_hb)
cmd.auto_arg[0]['hb'] = cmd.auto_arg[0]['delete']

# The code below is adapted from an early version of pyDSSP by Dr. Shintaro Minami
# https://github.com/ShintaroMinami/PyDSSP
# start pyDSSP 

CONST_Q1Q2 = 0.084
CONST_F = 332
DEFAULT_CUTOFF = -0.5
DEFAULT_MARGIN = 1.0

C8_ALPHABET = ['G', 'H', 'I', 'E', ' ']
C3_ALPHABET = ['L', 'H', 'E']


def get_hydrogen_atom_position(coord: torch.Tensor) -> torch.Tensor:
    # A little bit lazy (but should be OK) definition of H position here.
    vec_cn = coord[:, 1:, 0] - coord[:, :-1, 2]
    vec_cn = vec_cn / torch.linalg.norm(vec_cn, dim=-1, keepdim=True)
    vec_can = coord[:, 1:, 0] - coord[:, 1:, 1]
    vec_can = vec_can / torch.linalg.norm(vec_can, dim=-1, keepdim=True)
    vec_nh = vec_cn + vec_can
    vec_nh = vec_nh / torch.linalg.norm(vec_nh, dim=-1, keepdim=True)
    return coord[:, 1:, 0] + 1.01 * vec_nh


def get_hbond_map(
        coord: torch.Tensor,
        cutoff: float = DEFAULT_CUTOFF,
        margin: float = DEFAULT_MARGIN,
        return_e: bool = False
) -> torch.Tensor:
    # check input
    org_shape = coord.shape
    assert (len(org_shape) == 3) or (
                len(org_shape) == 4), "Shape of input tensor should be [batch, L, atom, xyz] or [L, atom, xyz]"
    coord = coord.unsqueeze(0) if len(org_shape) == 3 else coord
    b, l, a, _ = coord.shape
    # add pseudo-H atom if not available
    assert (a == 4) or (a == 5), "Number of atoms should be 4 (N,CA,C,O) or 5 (N,CA,C,O,H)"
    h = coord[:, 1:, 4] if a == 5 else get_hydrogen_atom_position(coord)
    # distance matrix
    nmap = repeat(coord[:, 1:, 0], '... m c -> ... m n c', n=l - 1)
    hmap = repeat(h, '... m c -> ... m n c', n=l - 1)
    cmap = repeat(coord[:, 0:-1, 2], '... n c -> ... m n c', m=l - 1)
    omap = repeat(coord[:, 0:-1, 3], '... n c -> ... m n c', m=l - 1)
    d_on = torch.linalg.norm(omap - nmap, dim=-1)
    d_ch = torch.linalg.norm(cmap - hmap, dim=-1)
    d_oh = torch.linalg.norm(omap - hmap, dim=-1)
    d_cn = torch.linalg.norm(cmap - nmap, dim=-1)
    # electrostatic interaction energy
    e = torch.nn.functional.pad(CONST_Q1Q2 * (1. / d_on + 1. / d_ch - 1. / d_oh - 1. / d_cn) * CONST_F, [0, 1, 1, 0])
    if return_e: return e
    # mask for local pairs (i,i), (i,i+1), (i,i+2)
    local_mask = ~torch.eye(l, dtype=bool)
    local_mask *= ~torch.diag(torch.ones(l - 1, dtype=bool), diagonal=-1)
    local_mask *= ~torch.diag(torch.ones(l - 2, dtype=bool), diagonal=-2)
    # hydrogen bond map (continuous value extension of original definition)
    hbond_map = torch.clamp(cutoff - margin - e, min=-margin, max=margin)
    hbond_map = (torch.sin(hbond_map / margin * torch.pi / 2) + 1.) / 2
    hbond_map = hbond_map * repeat(local_mask.to(hbond_map.device), 'l1 l2 -> b l1 l2', b=b)
    # return h-bond map
    hbond_map = hbond_map.squeeze() if len(org_shape) == 3 else hbond_map
    return hbond_map, h


def assign(coord: torch.Tensor, return_C8: bool = False):
    hbmap, _ = get_hbond_map(coord)
    hbmap = hbmap.transpose(-2, -1)  # convert into "i:C=O, j:N-H" form
    # identify turn 3, 4, 5
    turn3 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=3) > 0.
    turn4 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=4) > 0.
    turn5 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=5) > 0.
    # assignment of helical sses
    h3 = torch.nn.functional.pad(turn3[:, :-1] * turn3[:, 1:], [1, 3])
    h4 = torch.nn.functional.pad(turn4[:, :-1] * turn4[:, 1:], [1, 4])
    h5 = torch.nn.functional.pad(turn5[:, :-1] * turn5[:, 1:], [1, 5])
    # helix4 first
    helix4 = h4 + torch.roll(h4, 1, 1) + torch.roll(h4, 2, 1) + torch.roll(h4, 3, 1)
    h3 = h3 * ~torch.roll(helix4, -1, 1) * ~helix4
    h5 = h5 * ~torch.roll(helix4, -1, 1) * ~helix4
    helix3 = h3 + torch.roll(h3, 1, 1) + torch.roll(h3, 2, 1)
    helix5 = h5 + torch.roll(h5, 1, 1) + torch.roll(h5, 2, 1) + torch.roll(h5, 3, 1) + torch.roll(h5, 4, 1)
    # identify bridge
    unfoldmap = hbmap.unfold(-2, 3, 1).unfold(-2, 3, 1) > 0.
    unfoldmap_rev = unfoldmap.transpose(-4, -3)
    p_bridge = (unfoldmap[:, :, :, 0, 1] * unfoldmap_rev[:, :, :, 1, 2]) + (
                unfoldmap_rev[:, :, :, 0, 1] * unfoldmap[:, :, :, 1, 2])
    p_bridge = torch.nn.functional.pad(p_bridge, [1, 1, 1, 1])
    a_bridge = (unfoldmap[:, :, :, 1, 1] * unfoldmap_rev[:, :, :, 1, 1]) + (
                unfoldmap[:, :, :, 0, 2] * unfoldmap_rev[:, :, :, 0, 2])
    a_bridge = torch.nn.functional.pad(a_bridge, [1, 1, 1, 1])
    # ladder
    ladder = (p_bridge + a_bridge).sum(-1) > 0
    # H, E, L of C3
    H = (helix3 + helix4 + helix5) > 0
    E = ladder
    L = (~H * ~E)
    dssp_C3 = torch.stack([L, H, E], dim=-1)
    if return_C8 == False:
        return dssp_C3
    # C8
    assert return_C8 == False, 'C8 output is not yet implemented.'
    h3 = '+'.join([str(i.item() + 1) for i in (dssp[0, :, 0] == True).nonzero()])
    h4 = '+'.join([str(i.item() + 1) for i in (dssp[0, :, 1] == True).nonzero()])
    ss = '+'.join([str(i.item() + 1) for i in (dssp[0, :, 3] == True).nonzero()])
    dssp = torch.cat([dssp, (dssp.sum(-1) == 0).unsqueeze(-1)], dim=-1)
    return dssp

# end pyDSSP 
