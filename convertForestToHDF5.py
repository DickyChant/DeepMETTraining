#!/usr/bin/env python

import sys
import uproot
import numpy as np
import h5py
import os
import optparse


def deltaR(eta1, phi1, eta2, phi2):
    """ calculate deltaR """
    dphi = (phi1-phi2)
    while dphi >  np.pi: dphi -= 2*np.pi
    while dphi < -np.pi: dphi += 2*np.pi
    deta = eta1-eta2
    return np.hypot(deta, dphi)

# configuration
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-i', '--input', dest='input', help='input HiForest file', default='', type='string')
parser.add_option('-o', '--output', dest='output', help='output h5 file', default='', type='string')
parser.add_option("-N", "--maxevents", dest='maxevents', help='max number of events', default=-1, type='int')
(opt, args) = parser.parse_args()

if opt.input == '' or opt.output == '':
    sys.exit('Need to specify input and output files!')

# open HiForest file
upfile = uproot.open(opt.input)
print(upfile.keys())
# relevant trees
tree_evt = upfile['hiEvtAnalyzer/HiTree']
tree_pf  = upfile['particleFlowAnalyser/pftree']
tree_mu  = upfile['muonAnalyzer/MuonTree']

# setup arrays
maxNPF = 4500
nFeatures = 14

nEntries = tree_evt.num_entries if opt.maxevents == -1 else opt.maxevents

X    = np.zeros((nEntries, maxNPF, nFeatures), dtype=float)
XLep = np.zeros((nEntries, 2, nFeatures), dtype=float)
Y    = np.zeros((nEntries, 2), dtype=float)
EVT  = np.zeros((nEntries, 4), dtype=float)  # hiBin, hiNtracks, vz, evt

from tqdm import trange
# loop
for e in trange(nEntries):

    # muons
    Leptons = []
    nMu = tree_mu['nMu'].array()[e]
    muPt  = tree_mu['muPt'].array()[e]
    muEta = tree_mu['muEta'].array()[e]
    muPhi = tree_mu['muPhi'].array()[e]

    for i in range(min(2, nMu)):
        Leptons.append(( muPt[i], muEta[i], muPhi[i] ))

    ipf, ilep = 0, 0
    nPF = tree_pf['nPFpart'].array()[e]
    pfPt     = tree_pf['pfPt'].array()[e]
    pfEta    = tree_pf['pfEta'].array()[e]
    pfPhi    = tree_pf['pfPhi'].array()[e]
    pfId     = tree_pf['pfId'].array()[e]
    pfCharge = tree_pf['pfCharge'].array()[e]
    pfEcal   = tree_pf['pfEcal'].array()[e]

