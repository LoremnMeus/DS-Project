import numpy as np
import torch
import dgl
from dgl import DGLGraph
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc

#辅助函数
def one_of_k_encoding_unk(x, allowable_set):
    '将x与allowable_set逐个比较，相同为True， 不同为False, 都不同则认为是最后一个相同'
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_features(atom):
    possible_atom = ['C', 'N', 'O', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'S', 'Fe', 'Mg', 'B', 'DU'] #DU代表其他原子
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atom)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 1])
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), 
                                           [Chem.rdchem.HybridizationType.SP, 
                                            Chem.rdchem.HybridizationType.SP2,
                                            Chem.rdchem.HybridizationType.SP3, 
                                            Chem.rdchem.HybridizationType.SP3D])       
    return np.array(atom_features)

def get_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(bond_feats)

def gen_graph(molecule_smiles):
    G = DGLGraph()
    #加载smile生成mol对象
    molecule = Chem.MolFromSmiles(molecule_smiles)

    G.add_nodes(molecule.GetNumAtoms())

    node_features = []
    edge_features = []

    for i in range(molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i) 
        atom_i_features = get_atom_features(atom_i) 
        node_features.append(atom_i_features)
        
        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                G.add_edges(i,j) 
                bond_features_ij = get_bond_features(bond_ij) 
                edge_features.append(bond_features_ij)

    G.ndata['x'] = torch.from_numpy(np.array(node_features)) #dgl添加原子/节点特征
    G.edata['w'] = torch.from_numpy(np.array(edge_features)) #dgl添加键/边特征

    return G

from collections import namedtuple
from  rdkit  import  Chem 
from  rdkit.Chem  import  Draw 
from  rdkit.Chem.Draw  import  IPythonConsole 
from  rdkit.Chem  import  rdChemReactions 
import  rdkit 
from rdkit.Chem import AllChem
#from PIL.Image import Image

AtomInfo = namedtuple('AtomInfo',('mapnum','reactant','reactantAtom','product','productAtom'))
def map_reacting_atoms_to_products(rxn,reactingAtoms):
    ''' figures out which atoms in the products each mapped atom in the reactants maps to '''
    res = []
    for ridx,reacting in enumerate(reactingAtoms):
        reactant = rxn.GetReactantTemplate(ridx)
        for raidx in reacting:
            mapnum = reactant.GetAtomWithIdx(raidx).GetAtomMapNum()
            foundit=False
            for pidx,product in enumerate(rxn.GetProducts()):
                for paidx,patom in enumerate(product.GetAtoms()):
                    if patom.GetAtomMapNum()==mapnum:
                        res.append(AtomInfo(mapnum,ridx,raidx,pidx,paidx))
                        foundit = True
                        break
                    if foundit:
                        break
    return res
def get_mapped_neighbors(atom):
    ''' test all mapped neighbors of a mapped atom'''
    res = {}
    amap = atom.GetAtomMapNum()
    if not amap:
        return res
    for nbr in atom.GetNeighbors():
        nmap = nbr.GetAtomMapNum()
        if nmap:
            if amap>nmap:
                res[(nmap,amap)] = (atom.GetIdx(),nbr.GetIdx())
            else:
                res[(amap,nmap)] = (nbr.GetIdx(),atom.GetIdx())
    return res

BondInfo = namedtuple('BondInfo',('product','productAtoms','productBond','status'))
def find_modifications_in_products(rxn):
    ''' returns a 2-tuple with the modified atoms and bonds from the reaction '''
    reactingAtoms = rxn.GetReactingAtoms()
    amap = map_reacting_atoms_to_products(rxn,reactingAtoms)
    res = []
    seen = set()
    # this is all driven from the list of reacting atoms:
    for _,ridx,raidx,pidx,paidx in amap:
        reactant = rxn.GetReactantTemplate(ridx)
        ratom = reactant.GetAtomWithIdx(raidx)
        product = rxn.GetProductTemplate(pidx)
        patom = product.GetAtomWithIdx(paidx)

        rnbrs = get_mapped_neighbors(ratom)
        pnbrs = get_mapped_neighbors(patom)
        for tpl in pnbrs:
            pbond = product.GetBondBetweenAtoms(*pnbrs[tpl])
            if (pidx,pbond.GetIdx()) in seen:
                continue
            seen.add((pidx,pbond.GetIdx()))
            if not tpl in rnbrs:
                # new bond in product
                res.append(BondInfo(pidx,pnbrs[tpl],pbond.GetIdx(),'New'))
            else:
                # present in both reactants and products, check to see if it changed
                rbond = reactant.GetBondBetweenAtoms(*rnbrs[tpl])
                if rbond.GetBondType()!=pbond.GetBondType():
                    res.append(BondInfo(pidx,pnbrs[tpl],pbond.GetIdx(),'Changed'))
    return amap,res

def gen_graph_with_broken_bonds(molecule_smiles, reaction, template_label):
    rxn = rdChemReactions.ReactionFromSmarts(reaction)
    rxn.Initialize()
    atms, bnds = find_modifications_in_products(rxn)

    G = DGLGraph()
    #加载smile生成mol对象
    molecule = Chem.MolFromSmiles(molecule_smiles)

    G.add_nodes(molecule.GetNumAtoms())

    node_features = []
    edge_features = []
    edge_labels = []

    for i in range(molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i) 
        atom_i_features = get_atom_features(atom_i) 
        node_features.append(atom_i_features)
        
        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                G.add_edges(i,j) 
                bond_features_ij = get_bond_features(bond_ij) 
                edge_features.append(bond_features_ij)
                l = 0
                for b in bnds:
                    if (b.productAtoms[0] == i and b.productAtoms[1] == j) or (b.productAtoms[1] == i and b.productAtoms[0] == j):
                        l = template_label + 1
                edge_labels.append(l)

    G.ndata['x'] = torch.from_numpy(np.array(node_features)) #dgl添加原子/节点特征
    G.edata['w'] = torch.from_numpy(np.array(edge_features)) #dgl添加键/边特征
    G.edata['l'] = torch.from_numpy(np.array(edge_labels)) #dgl添加键/边特征

    return G