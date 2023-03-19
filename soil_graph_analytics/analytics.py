# File imports
from distribution import *
from utils import *
from particleSimulator import *

# Library imports
import scipy

#########################################
# FUNCTIONS TO ANALYZE PACKING FRACTION #
#########################################

def packing_fraction(soil_graph):
    """
    Given a soil graph that has been passed into particle_simulation(), return the packing fraction
    of the soil graph. That is, calculates the volume of the box containing the particles divided by 
    the exact volume of all the spheres.
    """
    # Define static variables
    exact_volume = soil_graph.exact_volume

    # Define dynamic variables
    highest_z = 0
    lowest_z = 0

    # Find box dimensions
    for particle in soil_graph.vertices.values():
        coordinates = particle['coordinates']
        size = particle['size']
        top_z, bottom_z = coordinates[2] + size , coordinates[2] - size
        if top_z > highest_z:
            highest_z = top_z
        if bottom_z < lowest_z:
            lowest_z = bottom_z
            
    simulation_volume = soil_graph.boundary_size**2*highest_z
    return exact_volume/simulation_volume

###########################################
# FUNCTIONS TO CREATE SLICE DISTRIBUTIONS #
###########################################

def create_slice_distribution(soil_graph, axis, slice_coordinate, num_bins):
    """
    Given a soil_graph that a simulation has been performed on, an axis to which the
    slice should be parallel to, and a coordinate to slice on that axis, returns a distribution
    of particles within that slice. 
    """
    # Specify which coordinate to query
    if axis == 'x':
        coord_index = 0
    if axis == 'y':
        coord_index = 1
    if axis == 'z':
        coord_index = 2

    # Create bins
    bin_size = 7/num_bins
    bin_number = 0
    bin_dict = {}
    while bin_number < num_bins:
        bin_dict[bin_size*(bin_number + 1)] = 0
        bin_number += 1

    # Set interval to check
    delta = 1000
    left_interval = slice_coordinate - delta
    right_interval = slice_coordinate + delta

    # Gather slice distribution
    for ix, particle in soil_graph.vertices.items():
        particle_location = particle['coordinates'][coord_index]
        size = particle['size']

        # If particle intersects with slice
        if particle_location + size > left_interval and particle_location - size < right_interval:
            log_size = math.log10(size) + 3 
            for bin in bin_dict.keys():
                if bin > log_size and bin - bin_size < log_size:
                    bin_dict[bin] += 1
    
    hist = list(bin_dict.values())
    bins = list(bin_dict.keys())

    # Normalize histogram
    hist = hist / (np.sum(hist) * bin_size)
    return hist, bins
 
if __name__ == '__main__':

    # Create simulation
    sample = generate_soil_sample(.2,.3)
    soil_graph = create_soil_graph(sample)
    particle_simulation(soil_graph)

    # Create distributions
    original_distribution = soil_graph.distribution
    binned = bin_distribution(original_distribution, 30)
    slice_distribution = create_slice_distribution(soil_graph, 'x', 3000, 30)

    # Visualize
    visualize_distribution(original_distribution)
    visualize_histogram(binned)
    visualize_histogram(slice_distribution)
    
"""
NOTES 

(2/26/23)
- Beginning by writing a function to find packing fraction of simulations

(2/27/23)
- Horizontal distributions
- graph traversal:

    - Maybe think about 

    - Sample randomly overtop layer
    - Search the dual graph
    - Traverse dual graph with weights as distance minimization.


(3/11/23 and 3/12/23)
- Creating Github Branch
- Creating functions to get slice distributions of soil simulations
- Comparing slice distributions to actual via KL divergence
- Figure out how to get KL divergence in code. 

PLAN:

- Write function to bin a continuous distribution
- Write function to do the following: Given a soil simulation, I want to be able to
query slices of it and check what the distribution of particles is. 
"""
