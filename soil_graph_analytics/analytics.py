# File imports
from distribution import realistic_distribution
from utils import *
from particleSimulator import *

####################################
# FUNCTIONS TO ANALYZE SIMULATIONS #
####################################

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
    # highest_x = 0

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

#


if __name__ == '__main__':
    total = 0
    sand = .3
    silt = .4
    # sample = generate_soil_sample(sand,silt)
    fracs = []
    # for i in range(50):
    #     soil_graph = create_soil_graph(sample)
    #     start = time.time()
    #     particle_simulation(soil_graph)
    #     end = time.time()
    #     print(end - start)
    #     frac = packing_fraction(soil_graph)
    #     fracs.append(frac)
    #     print(fracs)
    #     total += frac

    list = [0.6446316675087889, 0.6450923459278307, 0.5717659937520673, 0.6817558217981046, 0.6749418614461238, 0.6536435311159545, 0.6045882651743933, 0.6593549861196664, 0.6822788234292748, 0.6255628154153151, 0.613050148481514, 0.6444927166911357, 0.6969474278354546]
    mean = sum(list)/len(list)
    print(f'The mean packing fraction is {mean}')




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


(3/11/23)
- Creating Github Branch
- Creating functions to get slice distributions of soil simulations
- Comparing slice distributions to actual via KL divergence
- Figure out how to get KL divergence in code. 
"""
