from utils import *
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, weibull_min, weibull_max
import scipy.integrate as integrate
import random
import math
import numpy as np

np.seterr(all='ignore')
psd_linspace = np.array([0.375198,0.411878,0.452145,0.496347,0.544872,0.59814,0.656615,0.720807,0.791275,0.868632,0.953552,1.04677,1.14911,1.26145,1.38477,1.52015,1.66876,1.8319,2.011,2.2076,2.42342,2.66033,2.92042,3.20592,3.51934,3.8634,4.2411,4.65572,5.11087,5.61052,6.15902,6.76114,7.42212,8.14773,8.94427,9.81869,10.7786,11.8323,12.9891,14.2589,15.6529,17.1832,18.863,20.7071,22.7315,24.9538,27.3934,30.0714,33.0113,36.2385,39.7813,43.6704,47.9397,52.6264,57.7713,63.4192,69.6192,76.4253,83.8969,92.0988,101.103,110.987,121.837,133.748,146.824,161.177,176.935,194.232,213.221,234.066,256.948,282.068,309.644,339.916,373.147,409.626,449.672,493.633,541.892,594.869,653.025,716.866,786.949,863.883,948.338,1041.05,1142.83,1254.55,1377.2,1511.84,1659.64,1821.89,2000])
torch_psd_linspace=torch.tensor([0.375198,0.411878,0.452145,0.496347,0.544872,0.59814,0.656615,0.720807,0.791275,0.868632,0.953552,1.04677,1.14911,1.26145,1.38477,1.52015,1.66876,1.8319,2.011,2.2076,2.42342,2.66033,2.92042,3.20592,3.51934,3.8634,4.2411,4.65572,5.11087,5.61052,6.15902,6.76114,7.42212,8.14773,8.94427,9.81869,10.7786,11.8323,12.9891,14.2589,15.6529,17.1832,18.863,20.7071,22.7315,24.9538,27.3934,30.0714,33.0113,36.2385,39.7813,43.6704,47.9397,52.6264,57.7713,63.4192,69.6192,76.4253,83.8969,92.0988,101.103,110.987,121.837,133.748,146.824,161.177,176.935,194.232,213.221,234.066,256.948,282.068,309.644,339.916,373.147,409.626,449.672,493.633,541.892,594.869,653.025,716.866,786.949,863.883,948.338,1041.05,1142.83,1254.55,1377.2,1511.84,1659.64,1821.89,2000])
psd_logspace = np.logspace(2,2000,100)

#################
# ERROR CLASSES #
#################

class FractionError(Exception):
    pass

#####################################
# FUNCTIONS TO CREATE DISTRIBUTIONS #
#####################################

def generate_distribution(sandFrac, siltFrac):
    """
    Finds a distribution that matches soil composition.

    Input: 
        sandFrac: a float point number between 0 and 1
        siltFrac: a float point number between 0 and (1 - sandFrac)
    
    Returns: a continuous distribution function that matches the soil composition
    """
    # Verify input parameters
    if sandFrac + siltFrac > 1:
        raise FractionError('Sand fraction and silt fraction cannot sum to more than 1.')
    
    clayFrac = 1 - sandFrac - siltFrac

    # Define error bound
    epsilon = .01
    sandError = 1
    siltError = 1

    # Define statistical parameters (TO BE ITERATED OVER AND CHANGED)
    silt_shape_parameter = 1
    sand_shape_parameter = 1
    clay_shape_parameter = 1
    
    #  Randomly set shape parameters of Weibull's within ranges of particle sizes
    clay_scale_parameter = random.uniform(1,2.301)
    silt_scale_parameter = random.uniform(3.801,4.19)
    sand_scale_parameter = random.uniform(5.19,6.4)
    
    counter = 0

    while sandError > epsilon or siltError > epsilon:

        clay_scale_parameter = random.uniform(1,2.301)
        silt_scale_parameter = random.uniform(3.201,4.5)
        sand_scale_parameter = random.uniform(4.8, 6.4)
        clay_shape_parameter += 1
        silt_shape_parameter += 2
        sand_shape_parameter += 3

        # Define Weibull distribution
        sand_distribution = weibull_min(c = sand_shape_parameter, scale = sand_scale_parameter)
        silt_distribution = weibull_min(c = silt_shape_parameter, scale = silt_scale_parameter)
        clay_distribution = weibull_min(c = clay_shape_parameter, scale = clay_scale_parameter)

        # Define probability density function of combined distribution
        sandWeibull = lambda x: sand_distribution.pdf(x)
        siltWeibull = lambda x: silt_distribution.pdf(x)
        clayWeibull = lambda x: clay_distribution.pdf(x)

        # Make sure each individual distribution is mostly contained in domain
        sand_integral = integrate.quad(sandWeibull, 4.69, 6.301)
        silt_integral = integrate.quad(siltWeibull, 3.301, 4.69)
        clay_integral = integrate.quad(clayWeibull, 0, 3.301)

        # Combine individual distributions and scale by percent composition
        combined_distribution = lambda x: clayFrac * clay_distribution.pdf(x) +  siltFrac * silt_distribution.pdf(x) + sandFrac * sand_distribution.pdf(x) 

        # Redefine error terms
        sandError = abs(integrate.quad(combined_distribution, 4.69, 6.301)[0] - sandFrac)
        siltError = abs(integrate.quad(combined_distribution, 3.301, 4.69)[0] - siltFrac)

    return combined_distribution

def realistic_distribution(sandFrac, siltFrac):
    """
    Input: 
    sandFrac: a float point number between 0 and 1
    siltFrac: a float point number between 0 and (1 - sandFrac)
    
    Returns: a continuous distribution function that matches the soil composition that is
    more natural looking
    """
    distributions = []
    while len(distributions) < 5:
        distributions.append(generate_distribution(sandFrac, siltFrac))

    return lambda x: sum([(1/5) * distribution(x) for distribution in distributions])


########################################
# TOOLS FOR WORKING WITH DISTRIBUTIONS #
########################################

def bin_distribution(distribution, num_bins):
    """
    Given a continuous distribution and number of bins, returns a discrete distribution
    that bins the continuous distribution into the specified number of bins.
    """
    bin_size = 7/num_bins
    bin_number = 0
    hist = []
    bins = []
    while bin_number < num_bins:
        hist.append(integrate.quad(distribution, bin_size*bin_number, bin_size*(bin_number+ 1))[0])
        bins.append(bin_size*(bin_number + 1))
        bin_number += 1
    return hist, bins

def kl_divergence(distribution_1, distribution_2):
    """
    Given two discrete distributions, calculates the KL-divergence of the two distributions.
    """
    return sum(distribution_1[i] * math.log2(distribution_1[i]/distribution_2[i]) for i in range(len(distribution_1)))

def visualize_distribution(distribution):
    """
    Given a continuous distribution of the form created by realistic_distribution or generate_distribution, displays
    on a graph
    """
    x_vals = np.linspace(0,7,1000)
    combined_y = distribution(x_vals)
    plt.plot(x_vals, combined_y, color = 'green')
    plt.title('Particle Size Distribution')
    plt.xlabel('Log Particle Size (\u03BCm)', labelpad = 10)
    plt.ylabel('Percentage', labelpad = 10)
    plt.subplots_adjust(left = .2, bottom = .2)
    plt.show()

def visualize_histogram(binned_distribution):
    """
    Given a binned distribution of the form returned by binned_distribution(), plots
    histogram
    """
    # Set plot parameters
    hist = binned_distribution[0]
    bins = binned_distribution[1]
    bin_width = list(np.diff(bins))
    bin_width.append(.35)

    # Plot the histogram
    plt.bar(bins, hist, width = bin_width)
    plt.title('Histogram of data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == "__main__" :

    # VISUALIZE DISTRIBUTION
    distribution = realistic_distribution(.5, .2)
    visualize_distribution(distribution)

    # sand_scale = random.uniform(5.19,5.801)
    # silt_scale = random.uniform(3.801,4.19)
    # clay_scale = random.uniform(1,2.301)

    # sand_distribution = weibull_min(c=1, scale = sand_scale)
    # silt_distribution = weibull_min(c=1, scale = silt_scale)
    # clay_distribution = weibull_min(c=1, scale = clay_scale)
    # sand_y = sand_distribution.pdf(x_vals)
    # silt_y = silt_distribution.pdf(x_vals)
    # clay_y = clay_distribution.pdf(x_vals)

    # print(sand_scale, silt_scale, clay_scale)

    # plt.show()
    # plt.plot(x_vals, silt_y)
    # plt.show()
    # plt.plot(x_vals, clay_y)
    # plt.show()
    

    # RANGES (working in log nanometers)
    # 0 - 3.301 is clay, 3.301- 4.69 is silt, 4.69 - 6.301 is sand 
