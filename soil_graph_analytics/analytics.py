# File imports
from distribution import *
from utils import *
from particleSimulator import *


# Library imports
import scipy

##############################
# ANALYZING PACKING FRACTION #
##############################

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

#######################
# SLICE DISTRIBUTIONS #
#######################

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

#####################
# PERCOLATION TESTS #
#####################

def create_traversal_graph(soil_graph):
    """
    Given a soil graph that has had a simulation performed on it, create a new graph 
    suited for graph traversals. This new graph will have nodes as the contact points 
    between vertices, and edges TBD
    """
    traversal_graph = TraversalGraph(soil_graph)
    return traversal_graph

def get_percolation_rate(soil_graph):
    """
    Given a traversal graph of the form returned by create_traversal_graph(), returns
    a percolation rate which is calculate by traversing the graph.
    """
    traversal_graph = create_traversal_graph(soil_graph)
    # Get highest-z contact point
    max_z = [0, None]
    for contact_point_index, contact_point_info in traversal_graph.vertices.items():
        current_z = contact_point_info['contact_location'][-1]
        if max_z[0] < current_z:
            max_z[0] = current_z
            max_z[1] = contact_point_index
    
    total_percolation_time = 0
    current = max_z[1]
    counter = 0
    while True:
        if counter >10000:
            break
        counter += 1
        current_obj = traversal_graph.vertices[current]

        # Find first neighbor that goes downward
        next = None
        for neighbor in current_obj['neighbors']:
            if traversal_graph.vertices[neighbor]['contact_location'][-1] < current_obj['contact_location'][-1]:
                next = neighbor
        if next == None:
            break
        else:
            # Calculate percolation time
            dist = [(i - j)**2 for (i,j) in zip(current_obj['contact_location'], traversal_graph.vertices[neighbor]['contact_location'])]
            perc_time = math.sqrt(sum(dist))
            total_percolation_time += perc_time

        current = next

    return total_percolation_time/max_z[0]

#####################
# STATISTICAL TESTS #
#####################


if __name__ == '__main__':

    # Create simulation
    for i in range(1000):
    
        # GENERATE RANDOM FRACTION
        one = random.uniform(0,1)
        two = random.uniform(0, 1 - one)
        three = 1 - two - one
        choice_list = [one, two, three]
        sandFrac = random.choice(choice_list)
        choice_list.remove(sandFrac)
        siltFrac = random.choice(choice_list)

        # GENERATE SEED
        for j in range(10):

            sample = generate_soil_sample(sandFrac,siltFrac)
            soil_graph = create_soil_graph(sample)
            particle_simulation(soil_graph)

            # GET PERCOLATION RATE AND WRITE TO PERCOLATION.TXT
            traversal_graph = create_traversal_graph(soil_graph)
            print('im here')
            percolation_rate = get_percolation_rate(soil_graph)

            with open('percolation.txt', 'a') as f:
                f.write(f'[{sandFrac}, {siltFrac}, {percolation_rate}]')

            print('ive come out')


    # Create distributions
    # original_distribution = soil_graph.distribution
    # binned = bin_distribution(original_distribution, 30)
    # slice_distribution = create_slice_distribution(soil_graph, 'x', 3000, 30)

    # Visualize
    # visualize_distribution(original_distribution)
    # visualize_histogram(binned)
    # visualize_histogram(slice_distribution)


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

    (3/19/23)

    PLAN:

    Write function to depth-first search simulations
    What I wants:
    - Create another graph structure from the one had already, with contact points as nodes
    - Think about how to connect contact points

    NEXT TIME:
    - think about ways to weight the edges so that the traversal works
    - A way to visualize the traversal through the graph.

    PLAN for traversal (3.30.23)
    """
