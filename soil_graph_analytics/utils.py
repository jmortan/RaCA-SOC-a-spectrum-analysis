import matplotlib.pyplot as plt
import numpy as np
import csv
from samternary.ternary import Ternary

import torch
import torch .nn as nn
import torch .optim as optim

def plot_ternary_phase_diagram(sand,clay,val=None) :
    # create the figure and the two sets of axes             
    fig, ax_trans = plt.subplots(1,
                            figsize=[16,11.2])

    # transform ax_trans to ternary-plot style, which includes              
    # building axes and labeling the axes                                   
    cob = Ternary(ax_trans, bottom_name = '% Sand', left_name = '% Silt',
                  right_name = '% Clay',labelpad=20,fontsize=25)
    ax_trans.set_title("Soil Ternary Phase Diagram",fontsize=30)

    # use change of bases method within Ternary() to                        
    points = cob.B1_to_B2(sand,clay)

    # affine transform x,y points to ternary-plot basis                     
    if val is not None :
        cs = ax_trans.tricontourf(points[0],points[1],val)
    cs = ax_trans.scatter(points[0],points[1], color='r')

    # add color bar
    cbar = fig.colorbar(cs,ax=ax_trans,shrink=1,label="value")
    fig.subplots_adjust(bottom=0.2,hspace=0.01)

    # align axes to match USDA ternary phase diagram
    plt.xlim([1,0])
    
    # show plot
    plt.show()
    
    return fig, ax_trans, plt


def get_psd_from_list(alopsd, aloa) :
    return np.sum(alopsd * aloa, axis=0)

    # Magic data for fake psd generation
fake_psd_linspace=np.array([0.375198,0.411878,0.452145,0.496347,0.544872,0.59814,0.656615,0.720807,0.791275,0.868632,0.953552,1.04677,1.14911,1.26145,1.38477,1.52015,1.66876,1.8319,2.011,2.2076,2.42342,2.66033,2.92042,3.20592,3.51934,3.8634,4.2411,4.65572,5.11087,5.61052,6.15902,6.76114,7.42212,8.14773,8.94427,9.81869,10.7786,11.8323,12.9891,14.2589,15.6529,17.1832,18.863,20.7071,22.7315,24.9538,27.3934,30.0714,33.0113,36.2385,39.7813,43.6704,47.9397,52.6264,57.7713,63.4192,69.6192,76.4253,83.8969,92.0988,101.103,110.987,121.837,133.748,146.824,161.177,176.935,194.232,213.221,234.066,256.948,282.068,309.644,339.916,373.147,409.626,449.672,493.633,541.892,594.869,653.025,716.866,786.949,863.883,948.338,1041.05,1142.83,1254.55,1377.2,1511.84,1659.64,1821.89,2000])
torch_psd_linspace=torch.tensor([0.375198,0.411878,0.452145,0.496347,0.544872,0.59814,0.656615,0.720807,0.791275,0.868632,0.953552,1.04677,1.14911,1.26145,1.38477,1.52015,1.66876,1.8319,2.011,2.2076,2.42342,2.66033,2.92042,3.20592,3.51934,3.8634,4.2411,4.65572,5.11087,5.61052,6.15902,6.76114,7.42212,8.14773,8.94427,9.81869,10.7786,11.8323,12.9891,14.2589,15.6529,17.1832,18.863,20.7071,22.7315,24.9538,27.3934,30.0714,33.0113,36.2385,39.7813,43.6704,47.9397,52.6264,57.7713,63.4192,69.6192,76.4253,83.8969,92.0988,101.103,110.987,121.837,133.748,146.824,161.177,176.935,194.232,213.221,234.066,256.948,282.068,309.644,339.916,373.147,409.626,449.672,493.633,541.892,594.869,653.025,716.866,786.949,863.883,948.338,1041.05,1142.83,1254.55,1377.2,1511.84,1659.64,1821.89,2000])

# Generates a fake particle size distribution with the input fractions (by volume)
# of sand and silt.
def generate_fake_psd(linarr,fracSand,fracSilt) :
    arrSand = (linarr > 50.0).to(torch.float64)
    if torch.sum(arrSand,0) > 0 :
        arrSand = arrSand / torch.sum(arrSand[:-1]*np.diff(linarr),0) * fracSand

    arrSilt = ((linarr <= 50.0).to(torch.float64) * (linarr > 2.0).to(torch.float64))
    if torch.sum(arrSilt,0) > 0 :
        arrSilt = arrSilt / torch.sum(arrSilt[:-1]*np.diff(linarr),0) * fracSilt
    
    arrClay = (linarr <= 2.0).to(torch.float64)
    if torch.sum(arrClay,0) > 0 :
        arrClay = arrClay / torch.sum(arrClay[:-1]*np.diff(linarr),0) * (1.0 - fracSand - fracSilt)
    
    return (arrSand + arrSilt + arrClay)

# Given a particle size distribution over some linear space, return its fraction
# by volume of sand, silt, and clay
def get_psd_fracs(psd, linarr) :
    arrSand = (linarr >  50.0).to(torch.float64)
    arrSilt = (linarr <= 50.0).to(torch.float64) * (linarr > 2.0).to(torch.float64)
    arrClay = (linarr <=  2.0).to(torch.float64)

    frcSand = torch.sum(psd[:-1] * arrSand[:-1] * torch.diff(linarr))
    frcSilt = torch.sum(psd[:-1] * arrSilt[:-1] * torch.diff(linarr))
    frcClay = torch.sum(psd[:-1] * arrClay[:-1] * torch.diff(linarr))
    
    return frcSand, frcSilt, frcClay

def get_psd_from_file(path) :
    with open(path, 'r', encoding='utf8') as psd_file :
        scrape_on = False
        skiplines = 0
        
        tbinsize = []
        tdvols = []
        tdsa = []
        tdn = []
        
        
        for line in psd_file :
            spl = line.split("\t")
            if spl[0] == "Channel Diameter (Lower)" : 
                scrape_on = True
                continue
            if scrape_on : skiplines += 1
            if not scrape_on or skiplines < 3 : continue
            
            if spl[0] == "2000" : 
                tbinsize = tbinsize + [float(2000)]
                break
            
            tbinsize = tbinsize + [float(spl[0])]
            tdvols   = tdvols   + [float(spl[1])]
            tdsa     = tdsa     + [float(spl[2])]
            tdn      = tdn      + [float(spl[3])]
            
        return tbinsize, tdvols, tdsa, tdn

class LinearMixingRegressor(nn.Module):
    def __init__(self, alopsd):
        super().__init__()
        self.alopsd = alopsd
        self.abunds = nn.Parameter(torch.tensor([1.0/len(alopsd)]*len(alopsd)))
    def forward(self, y):
        pred = torch.sum(torch.transpose(torch.transpose(self.alopsd,0,1) * self.abunds,0,1), 0)
        
        # Add in a fake Lagrange multiplier to discourage abundances < 0.01 or > 0.99
        oobs = torch.sum(((-1.0 * (self.abunds < 0.01).float() + (self.abunds > 0.99).float()) * self.abunds)**2)
        
        return - torch.sum((y - pred) ** 2 / 2.) * (1 + 1000*oobs)

def regress_abundances_by_psd(psd,alocomp) :
    model = LinearMixingRegressor(alocomp)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10000) :
        optimizer.zero_grad()
        loss = model(psd)
        e = - torch.mean(loss)
        e.backward()
        optimizer.step()
    
    return model.abunds.detach().tolist(), loss.detach().item()

class AbundanceRegressor(nn.Module):
    def __init__(self, alopsd):
        super().__init__()
        self.alopsd = alopsd
        self.abunds = nn.Parameter(torch.tensor([1.0/len(alopsd)]*len(alopsd)))
    def forward(self, y):
        pred = torch.sum(torch.transpose(torch.transpose(self.alopsd,0,1) * self.abunds,0,1), 0)
        predSand, predSilt, predClay = get_psd_fracs(pred,torch_psd_linspace)
        return - ((y[0] - predSand)**2 + (y[1] - predSilt)**2 + (y[2] - predClay)**2)

def regress_abundances_by_fracs(fracSand,fracSilt,alocomp) :
    fracClay = 1.0 - fracSand - fracSilt;
    model = AbundanceRegressor(alocomp)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10000) :
        optimizer.zero_grad()
        loss = model([fracSand, fracSilt, fracClay])
        e = - torch.mean(loss)
        e.backward()
        optimizer.step()
    
    return model.abunds.detach().tolist(), loss.detach().item()

if __name__ == "__main__" :
    sand_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0,
              0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.05, 0.1])
    clay_test = np.array([0.9, 0.7, 0.5, 0.3, 0.1, 0.2, 0.1, 0.15, 0, 0.1,
              0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    val_test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0,
              1, 2])

# How to go from vector of y-values to workable data