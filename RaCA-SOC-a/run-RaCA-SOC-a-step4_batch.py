import import_ipynb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde

import torch
import torch .nn as nn
import torch .optim as optim

import json
import gc

from tqdm import tqdm

import pickle

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-m", "--min", dest="minIndex",
                  help="Minimum data index to fit")
parser.add_option("-o", "--max", dest="maxIndex",
                  help="Maximum data index to fit")

(options, args) = parser.parse_args()
minIndex = int(options.minIndex)
maxIndex = int(options.maxIndex)

plt.rcParams['text.usetex'] = True




data = np.loadtxt("../data/RaCA-spectra-raw.txt",
                 delimiter=",", dtype=str)

sample_soc   = data[minIndex:maxIndex,2162].astype('float32')

dataI = data[minIndex:maxIndex,1:2152].astype('float32')
XF = np.array([x for x in range(350,2501)]);

del data

def postProcessSpectrum(xin,xout,refin) :
    return np.interp(xout, xin, refin)

for iSpec in tqdm(range(dataI.shape[0])) :
            
    wavelengths = [x for x in range(350,2501)]
    reflectance = dataI[iSpec,:]
    
    newwave = np.array([wavelengths[i] for i in range(len(wavelengths)) if reflectance[i] is not None and reflectance[i] > 0.0 and reflectance[i] <= 1.0])
    newref  = np.array([reflectance[i] for i in range(len(reflectance)) if reflectance[i] is not None and reflectance[i] > 0.0 and reflectance[i] <= 1.0])
    
    dataI[iSpec,:] = postProcessSpectrum(newwave,XF,newref)
    
    
#######
KEndmembers = 90
NPoints = 1
NData = dataI.shape[0]
MSpectra = 2151

# load JSON file with pure spectra
endMemMap = json.load(open('../data/endmember spectral data.json'))

# get reflectance spectra (y axis) and wavelength grid (x axis)
endMemList = [x for x in endMemMap.keys()];
endMemList.remove("General")
XF = endMemMap["General"]["Postprocessed Wavelength Axis [nm]"]
F = [endMemMap[x]["Postprocessed Reflectance"] for x in endMemList]

# get density, radius info and merge into relevant arrays
rhos = [endMemMap[x]["Density (Mg/m^3)"] for x in endMemList]
rads = [endMemMap[x]["Effective Radius (nm)"] for x in endMemList]

class LinearMixingModel(nn.Module):
    def __init__(self, seedFs, seedFsoc, seedMs, rhorad, seedrrsoc, nepochs):
        super().__init__()
        # fixed quantities
        self.rhorad = rhorad
        self.fs     = seedFs
        
        # model parameters
        self.fsoc   = nn.Parameter(seedFsoc)
        self.rrsoc  = nn.Parameter(seedrrsoc)
        self.ms     = nn.Parameter(seedMs)
        
        # model output
        self.Ihat   = 0;
        
        # variables for tracking optimization
        self.epoch = 0;
        self.nepochs = nepochs;
        
        self.lsq = np.zeros(nepochs);
        self.loss = np.zeros(nepochs);
        self.bdsALoss = np.zeros(nepochs);
        self.bdsFLoss = np.zeros(nepochs);
        self.omrsLoss = np.zeros(nepochs);
        self.diffloss1 = np.zeros(nepochs);
        self.difflossfull = np.zeros(nepochs);
        
        
    def forward(self, y):
        msocs,Is,Imax = y
        rrFull    = torch.cat((self.rhorad,self.rrsoc))
        mFull     = torch.cat((self.ms,msocs.unsqueeze(1)),dim=1)
        mFull     = (mFull.t() / torch.sum(mFull,axis=1)).t()
        fFull     = torch.cat((self.fs,self.fsoc.unsqueeze(0)),dim=0)
        self.Ihat = torch.matmul(torchA(mFull,rrFull).float(),fFull.float())
                
        # Add in a fake Lagrange multiplier to discourage abundances < 0.001 or > 0.999
        oobsA = torch.sum((mFull < 0.001).float() * (mFull - 0.001)**2) 
        oobsA = oobsA + torch.sum((mFull > 0.999).float() * (mFull + 0.001 - 1.0) **2)

        # Add in a fake Lagrange multiplier to discourage Fsoc < 0 and Fsoc > 1
        oobsF = 1.0 * torch.sum((self.fsoc < 0.0).float() * (self.fsoc ** 2)) 
        oobsF = oobsF + 1.0 * torch.sum((self.fsoc > 1.0).float() * (1.0 - self.fsoc) **2)
        
        # Add in 1st derivative loss to smooth the curves
        diffloss = torch.sum(torch.diff(self.fsoc) ** 2)
        self.diffloss1[self.epoch] = diffloss.detach().item();
        
        diffloss += torch.sum(torch.diff(torch.diff(self.fsoc)) ** 2)
        
        # Compute the loss function, which is the mean-squared error between data and prediction,
        # with a multiplicative factor for our fake Lagrange multipliers
        lsq = torch.sum((Is - self.Ihat) ** 2)
        loss = lsq * (1 + 100.0* diffloss + 100.0*oobsA + 1000.0*oobsF) # + 10000.0*omrs
        
        # Report optimization statistics
        self.lsq[self.epoch]  = lsq.detach().item()
        self.loss[self.epoch] = loss.detach().item();
        self.bdsALoss[self.epoch] = oobsA.detach().item();
        self.bdsFLoss[self.epoch] = oobsF.detach().item();
        self.difflossfull[self.epoch] = diffloss.detach().item();
        
        self.epoch += 1;
        
        return loss


rrs=[]
socspecs = np.zeros([MSpectra,1501])
ms = np.zeros([100,KEndmembers-1,1501])

offset=0

with open('results/step3/step2_systematics_N100_NpAll_E100k_init.pkl', 'rb') as file:
        (model,optimizer,F,seedFsoc,trueFsoc,seedMs,dataIndices,msoc,rhorads,seedSOCrr,trueSOCrr) = pickle.load(file)
        rrs += [model.rrsoc.detach().item()]
        socspecs[:,0] = np.array(model.fsoc.tolist())

        tcorrms = np.array(model.ms.tolist())
        tcorrms = (tcorrms > 0.0).astype('float32') * tcorrms
        tcorrms = (tcorrms.T / (np.sum(tcorrms,axis=1)) * (1 - msoc)).T 

        ms[:,:,0] = tcorrms

        print(model.rrsoc.detach().item())

for i in range(1500) :

    with open('results/step3/step2_systematics_N100_NpAll_E100k_%i.pkl'%(i+offset), 'rb') as file:
        (model,F,seedFsoc,trueFsoc,seedMs,dataIndices,msoc,rhorads,seedSOCrr,trueSOCrr) = pickle.load(file)

        rrs += [model.rrsoc.detach().item()]

        socspecs[:,i+1] = np.array(model.fsoc.tolist())
        if np.sum(np.isnan(socspecs[:,i+1]).astype('float32')) > 0 :
            print("Spectrum ",i,"is NaN.")

        tcorrms = np.array(model.ms.tolist())
        tcorrms = (tcorrms > 0.0).astype('float32') * tcorrms
        tcorrms = (tcorrms.T / (np.sum(tcorrms,axis=1)) * (1-msoc)).T 

        ms[:,:,i+1] = tcorrms

socspecs[:,861]  = socspecs[:,862]
socspecs[:,1046] = socspecs[:,1047]
rrs[861]   = rrs[862]
rrs[1046]  = rrs[1047]
ms[:,:,861]  = ms[:,:,862]
ms[:,:,1046] = ms[:,:,1047]

def getSeedMs() :

    fsoc = np.mean(socspecs,axis=1).tolist()
    F = [endMemMap[x]["Postprocessed Reflectance"] for x in endMemList]
    F = np.array(F + [fsoc])
    fsoc = np.array(fsoc)

    seedrrsoc = np.mean(np.array(rrs))
    seedMs    = np.mean(ms,axis=(0,2)) # covers all seedMs but msoc
    seedMsoc  = np.mean(sample_soc)/100.0 
    seedMs = np.append(seedMs,seedMsoc)
    
    return seedMs, seedrrsoc, seedMsoc

seedMs, seedrrsoc, seedMsoc = getSeedMs()

dataIndices = np.random.choice(NData,NPoints,replace=False)
msoc = sample_soc[dataIndices]/100.0

def A(ms,rhorads) :
    tA = ms / rhorads
    return (tA.T / np.sum(tA)).T

def torchA(ms,rhorads) :
    tA = ms / rhorads
    return (tA.t() / torch.sum(tA)).t()

rhorads = np.array(rhos)*np.array(rads)
seedAs = A(seedMs,np.append(rhorads,seedrrsoc))

class LinearMixingSOCPredictor(nn.Module):
    def __init__(self, seedFs, seedMs, trueMsoc, rhorad, seedrrsoc, nepochs):
        super().__init__()
        # fixed quantities
        self.rhorad = rhorad;
        self.fs     = seedFs;
        self.truemsoc = trueMsoc;
        
        # model parameters
        self.rrsoc  = nn.Parameter(seedrrsoc);
        self.ms     = nn.Parameter(seedMs);
        
        # model output
        self.Ihat   = 0;
        
        # variables for tracking optimization
        self.epoch = 0;
        self.nepochs = nepochs;
        
        self.lsq = np.zeros(nepochs);
        self.loss = np.zeros(nepochs);
        self.socbias = np.zeros(nepochs);
        self.bdsALoss = np.zeros(nepochs);
        self.diffloss1 = np.zeros(nepochs);
        self.difflossfull = np.zeros(nepochs);
        
        
    def forward(self, y):
        rrFull    = torch.cat((self.rhorad,self.rrsoc.unsqueeze(0)))
        mFull     = (self.ms.t() / torch.sum(self.ms)).t()
        self.Ihat = torch.matmul(torchA(mFull,rrFull).float(),self.fs.float())
                
        # Add in a fake Lagrange multiplier to discourage abundances < 0.001 or > 0.999
        oobsA = torch.sum((mFull < 0.001).float() * (mFull - 0.001)**2) 
        oobsA = oobsA + torch.sum((mFull > 0.999).float() * (mFull + 0.001 - 1.0) **2)
        
        # Add in 1st derivative loss to smooth the curves
        diffloss = torch.sum(torch.diff(self.Ihat) ** 2)
        self.diffloss1[self.epoch] = diffloss.detach().item();
        
        diffloss += torch.sum(torch.diff(torch.diff(self.Ihat)) ** 2)
        
        # Compute the loss function, which is the mean-squared error between data and prediction,
        # with a multiplicative factor for our fake Lagrange multipliers
        lsq = torch.sum((y - self.Ihat) ** 2)
        loss = lsq * (1 + 100.0* diffloss + 100.0*oobsA)
        
        # Report optimization statistics
        self.lsq[self.epoch]  = lsq.detach().item()
        self.loss[self.epoch] = loss.detach().item();
        self.socbias[self.epoch]  = self.truemsoc - mFull[-1];
        self.bdsALoss[self.epoch] = oobsA.detach().item();
        self.difflossfull[self.epoch] = diffloss.detach().item();
        
        self.epoch += 1;
        
        return loss
    
    

offset = 0

for i in tqdm(range(NData)) :
    
    # generate msoc
    dataIndices = i
    msoc = sample_soc[dataIndices]/100.0
    
    # generate new seed Fsoc
    seedMs, seedrrsoc, seedMsoc = getSeedMs()
    
    # seed data: A[1:,:] and initial F's
    tF       = torch.tensor(F.tolist())
    tseedMs  = torch.tensor(seedMs.tolist())
    tmsoc    = torch.tensor(msoc)
    trhorads = torch.tensor(rhorads.tolist())
    trrsoc   = torch.tensor(seedrrsoc)

    # empirical data: (SOC values, reflectances, and max normalized reflectance)
    ys = torch.tensor(dataI[dataIndices])

    nepochs = 20000
    model = LinearMixingSOCPredictor(tF, tseedMs, tmsoc, trhorads, trrsoc, nepochs)
    optimizer = optim.Adam(model.parameters(), lr = 0.000005, betas=(0.99,0.999))
    
    print("\t - Training model",i+offset+minIndex)
    for epoch in tqdm(range(nepochs)) :
        loss = model(ys)
        e = torch.mean(loss)
        e.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("Epoch ",epoch,": ", loss.detach().item(), model.lsq[-1], model.lsq[-1] / (0.05 ** 2) / (NPoints*MSpectra))
    
    with open('step4_predictions_N1_NpAll_E20k_batch_%i.pkl'%(i+offset+minIndex), 'wb') as file:
        pickle.dump((model,seedMs,seedrrsoc,dataIndices,msoc), file)
        