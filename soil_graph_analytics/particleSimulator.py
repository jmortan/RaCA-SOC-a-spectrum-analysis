# File imports
from distribution import realistic_distribution
from undirectedGraphs import *
from utils import *

# Other
import random
import time
import copy
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

#################
# ERROR CLASSES #
#################

class OverlapError(Exception):
    pass

############################
# GENERAL HELPER FUNCTIONS #
############################

def merge_lists(list_1, list_2):
    """
    Given two lists, where each list consists of tuples in the form (vertex_index, vertex_z_coordinate) sorted
    in decreasing z_coordinate order, merge the two lists into one list while maintaining the sort.
    """
    pointer_1 = 0
    pointer_2 = 0 
    stop_1 = len(list_1)
    stop_2 = len(list_2)
    if stop_1 == 0:
        return list_2
    elif stop_2 == 0:
        return list_1
    merged = []
    while pointer_1 < stop_1 and pointer_2 < stop_2:
        if list_1[pointer_1][1] > list_2[pointer_2][1]:
            merged.append(list_1[pointer_1])
            pointer_1 += 1
        elif list_1[pointer_1][1] < list_2[pointer_2][1]:
            merged.append(list_2[pointer_2])
            pointer_2 += 1
        elif list_1[pointer_1][1] == list_2[pointer_2][1]:
            pointer_2 += 1
    if pointer_1 == stop_1:
        merged += list_2[pointer_2:]
    if pointer_2 == stop_2:
        merged += list_1[pointer_1:]
    return merged

def increment_spherical_angle(cartesian, angle_increment):
    spherical = [np.linalg.norm(cartesian), np.arctan2(cartesian[1], cartesian[0]), np.arccos(cartesian[2] / np.linalg.norm(cartesian))]
    spherical[2] += angle_increment
    at_bottom = False

    # If incremented past pi radians
    if spherical[2] >= np.pi:
        spherical[2] = np.pi
        at_bottom = True
    
    x = spherical[0] * np.sin(spherical[2]) * np.cos(spherical[1])
    y = spherical[0] * np.sin(spherical[2]) * np.sin(spherical[1])
    z = spherical[0] * np.cos(spherical[2])

    return [x, y, z, at_bottom]


def rotate_triangle(vertices, index, radians):
    # Get the indices of the other two vertices
    other_indices = [i for i in vertices.keys() if i != index]

    # Calculate the vector representing the side opposite the chosen vertex
    side_vector = np.array(vertices[other_indices[1]]) - np.array(vertices[other_indices[0]])

    # Calculate the rotation axis and matrix
    axis = side_vector / np.linalg.norm(side_vector)
    rotation_matrix = np.array([[np.cos(radians) + axis[0]**2*(1-np.cos(radians)), 
                                    axis[0]*axis[1]*(1-np.cos(radians)) - axis[2]*np.sin(radians), 
                                    axis[0]*axis[2]*(1-np.cos(radians)) + axis[1]*np.sin(radians)], 
                                [axis[1]*axis[0]*(1-np.cos(radians)) + axis[2]*np.sin(radians), 
                                    np.cos(radians) + axis[1]**2*(1-np.cos(radians)), 
                                    axis[1]*axis[2]*(1-np.cos(radians)) - axis[0]*np.sin(radians)], 
                                [axis[2]*axis[0]*(1-np.cos(radians)) - axis[1]*np.sin(radians), 
                                    axis[2]*axis[1]*(1-np.cos(radians)) + axis[0]*np.sin(radians), 
                                    np.cos(radians) + axis[2]**2*(1-np.cos(radians))]])

    # Rotate the vertex
    rotated_vertex = np.array(vertices[index]) - np.array(vertices[other_indices[0]])
    rotated_vertex = np.dot(rotation_matrix, rotated_vertex)
    rotated_vertex = rotated_vertex + np.array(vertices[other_indices[0]])
    
    # Check if the rotated vertex intersects with other spheres
    for i in other_indices:
        distance = np.linalg.norm(np.array(vertices[i]) - rotated_vertex)
        if distance < 2:
            return vertices

    # Update the vertices dictionary with the rotated vertex
    vertices[index] = list(rotated_vertex)
    return vertices

########################################
# FUNCTIONS FOR SAMPLING DISTRIBUTIONS #
########################################

def generate_soil_sample(sandFrac, siltFrac):
    """
    Input: 
        sandFrac: float from 0-1
        siltFrac: float from 0 - (1-sandFrac)

    Returns: A list of tuples, where each tuple contains a particle size in (mm) and number of 
    particles of that size in the sample. Halts when total numbers of particles is above 1000.
    """
    distribution = realistic_distribution(sandFrac, siltFrac)
    sample = []
    total_particles = 0

    # Randomly sample distribution
    while total_particles < 2000:
        sample_x = random.uniform(0,6.301)
        size = 10 ** (sample_x - 3)
        frequency = round(distribution(sample_x) * 100)
        if frequency == 0:
            continue
        total_particles += frequency
        entry = [size, frequency]
        sample.append(entry)
    
    return sample, [distribution, (sandFrac, siltFrac)]

def test_sample(soil_sample):
    """
    Given a soil sample of the form returned by generate_soil_sample, returns composition of sample by fraction 
    of sand, silt, and clay
    """
    sand_total = 0
    silt_total = 0
    clay_total = 0
    
    for elt in soil_sample:
        if elt[0] >= 50:
            sand_total += elt[1]
        elif elt[0] >= 2 and elt[0] < 50:
            silt_total += elt[1]
        elif elt[0] < 2:
            clay_total += elt[1]

    total = sand_total + silt_total + clay_total
    return f'Composition: {round((clay_total/total), 3) * 100} percent clay, {round((silt_total/total), 3) * 100} percent silt, {round((sand_total/total), 3) * 100} percent sand'

################################################
# FUNCTIONS FOR WORKING WITH UNDIRECTED GRAPHS #
################################################

def create_soil_graph(soil_sample):
    """
    Given a soil sample of the form returned by generate_soil_sample, create a graph object
    of the particles    
    """

    sample_information = copy.deepcopy(soil_sample)
    sample = sample_information[0]
    distribution = sample_information[1][0]
    fractions = sample_information[1][1]
    soil_graph = Graph()

    # Add particles from sample to graph
    volume = 0 
    smallest_particle = 3000
    while sample:
        tuple_ix = random.randint(0,len(sample) - 1)
        current = sample[tuple_ix]
        size = current[0]
        if size <  smallest_particle:
            smallest_particle = size
        volume += 4/3 * math.pi * size ** 3

        # Add particle to graph as vertex with no neighbors or coordinates
        soil_graph.add_vertex(size)

        # Remove particle from sample
        if sample[tuple_ix][1] == 1:
            del sample[tuple_ix]
        else:
            sample[tuple_ix][1] -= 1
    
    # Set distribution of soil graph
    soil_graph.fractions = fractions
    soil_graph.distribution = distribution
    
    # Set smallest particle size
    soil_graph.smallest_particle = smallest_particle
    
    # Set exact volume
    soil_graph.exact_volume = volume

    # Find appropriate boundary size.
    soil_graph.boundary_size = round(np.cbrt(volume))

    # Create divisions for coordinate containers
    partition_length = round(smallest_particle * 10000, 2)
    soil_graph.partition_length = partition_length
    num_divisions = math.ceil(soil_graph.boundary_size / partition_length)
    soil_graph.num_divisions = num_divisions
    for i in [round(partition_length*i,2) for i in range(0,num_divisions)]:
        for j in [round(partition_length*i,2) for i in range(0,num_divisions)]:
            soil_graph.coordinate_containers[(i,j)] = []

    return soil_graph

###################################################
# HELPER FUNCTIONS FOR FINDING RELEVANT PARTICLES #
###################################################

def relevant_containers(soil_graph, particle_size, location):
    """
    Given a soil graph of the form returned by create_soil_graph, a particle size, and an xy-location, returns
    a list of xy-coordinate-containers to check.
    """
    partition_length = soil_graph.partition_length
    num_divisions = soil_graph.num_divisions - 1
    # Looking at particle from above, define left-most, right-most, up-most, down-most positions of circle
    left = location[0] - particle_size
    right = location[0] + particle_size
    up = location[1] + particle_size
    down = location[1] - particle_size
    # Iterate through containers to find which ones are relevant to particle
    lower_x = 0
    lower_y = 0
    greater_x = num_divisions * partition_length
    greater_y = num_divisions * partition_length
    for i in range(0,num_divisions):
        current = i * partition_length
        if current < left and current + partition_length > left:
            lower_x = current
        if current < right and current + partition_length > right:
            greater_x = current
        if current < down and current + partition_length > down:
            lower_y = current
        if current < up and current + partition_length > up:
            greater_y = current

    # Record relevant containers
    x_steps = int((greater_x - lower_x)/partition_length)
    y_steps = int((greater_y - lower_y)/partition_length)
    containers = []
    for i in range(x_steps + 1):
        for j in range(y_steps + 1):
            containers.append((round(lower_x + i * partition_length, 2), round(lower_y + j * partition_length, 2)))

    return containers

def find_first_relevant_particles(soil_graph, particle_size, location):
    """
    Given a soil graph of the form returned from create_soil_graph, a particle size and a location,
    returns a list of tuples with a vertex index and a z-coordinate representing
    relevant particles to the motion of the particle in question.
    """
    containers_to_check = relevant_containers(soil_graph, particle_size, location)
    relevant_particle_lists = []
    for container in containers_to_check:
        particles = soil_graph.coordinate_containers[container]
        if particles:
            relevant_particle_lists.append(particles)

    # Merge particle lists in decreasing z
    relevant_particles = []
    for i in range(len(relevant_particle_lists)):
        relevant_particles = merge_lists(relevant_particles, relevant_particle_lists[i])

    return relevant_particles

def surrounding_particles(soil_graph, dropped_particle_size, first_contacted_size, first_contacted_location):
    """
    Given a soil_graph, a dropped particle size, a first contacted size, and first contacted location, gives a list
    of all relevant particles in the form (particle index, top of sphere-z, bottom of sphere-z)
    """
    containers = relevant_containers(soil_graph, 2*dropped_particle_size + first_contacted_size, first_contacted_location)
    relevant = set()
    highest_z = first_contacted_location[2] + first_contacted_size + 2* dropped_particle_size
    lowest_z = first_contacted_location[2] - first_contacted_size - 2* dropped_particle_size
    for container in containers:
        particles = soil_graph.coordinate_containers[container]
        for particle in particles:
            if (particle[1] < highest_z and particle[1] > lowest_z) or (particle[2] > lowest_z and particle[2] < highest_z):
                relevant.add(particle[0])
    return relevant

##############################################
# HELPER FUNCTIONS FOR PARTICLE INTERACTIONS #
##############################################

def check_overlap(vertex_tuple_1, vertex_tuple_2):
    """
    Given two tuples, where each tuple contains two elements: in the 0th index a vertex size and in the 1st index a location,
    returns True if particles are overlapping and False otherwise. If the particle is in exact contact it returns True.
    """
    particle_1_size = vertex_tuple_1[0]
    particle_1_location = vertex_tuple_1[1]
    particle_2_size = vertex_tuple_2[0]
    particle_2_location = vertex_tuple_2[1]
    if math.dist(particle_1_location, particle_2_location) <= particle_1_size + particle_2_size:
        return True
  
def check_all_overlap(soil_graph):
    """
    Check if any particles in graph overlap and returns True if they do. 
    """
    for ix, info in soil_graph.vertices.items():
        for ix_2, info_2 in soil_graph.vertices.items():
            if info['coordinates'] != None and info_2['coordinates'] != None:
                if check_overlap((info['size'], info['coordinates']),(info_2['size'], info_2['coordinates'])):
                    return True
    return False

def check_contact(soil_graph, vertex_index_1, vertex_index_2):
    """
    Given a particle dictionary and two vertex indices, checks if ,
    vertices are in contact with each other.

    Returns True if vertices are in contact with each other. 
    Returns False if vertices are not in contact with each other or
    if one of the vertices has not been assigned a coordinate. 

    Raises Over
    lapError if two vertices are overlapping
    """
    smallest_particle_size = soil_graph.smallest_particle
    particle_dictionary = soil_graph.vertices
    particle_1_size = particle_dictionary[vertex_index_1]['size']
    particle_1_location = particle_dictionary[vertex_index_1]['coordinates']
    particle_2_size = particle_dictionary[vertex_index_2]['size']
    particle_2_location = particle_dictionary[vertex_index_2]['coordinates']
    
    if particle_1_location == None or particle_2_location == None:
        return False
    if abs(math.dist(particle_1_location, particle_2_location) - (particle_1_size + particle_2_size)) < (1/5) * smallest_particle_size:
        return True
    else:
        return False

def adjusted_location(soil_graph, vertex_tuple_1, vertex_tuple_2):
    """
    Given two tuples, where each tuple contains two elements: in the 0th index a vertex size and in the 1st index a location,
    changes the coordinates of the first vertex so the two vertices are "in contact" with each other. Only works if check_overlap
    returns true.
    """
    particle_1_size = vertex_tuple_1[0]
    particle_1_location = vertex_tuple_1[1]
    particle_2_size = vertex_tuple_2[0]
    particle_2_location = vertex_tuple_2[1]
    if .001 < (1/5)*soil_graph.smallest_particle:
        epsilon = .001
    else:
        epsilon = (1/5)*soil_graph.smallest_particle
    delta_z = particle_2_size
    going_up = True
    while abs(math.dist(particle_1_location, particle_2_location) - (particle_1_size + particle_2_size)) > epsilon:
        particle_1_location[2] += delta_z
        # Check if it should be moved up or down. If there is a switch, half delta_z and switch direction
        if check_overlap(vertex_tuple_1, vertex_tuple_2):
            if going_up == True:
                pass
            else:
                going_up = True
                delta_z *= (-1/2)
        else:
            if going_up == False:
                pass
            else:
                going_up = False
                delta_z *= (-1/2)
            
    return particle_1_location

def adjust_second_contact_location(soil_graph, particle_info, dropped_index):
    """
    Given a list of the sizes and locations of the dropped particle, the first contact, and the second contact, rotate the dropped particle so it 
    is in contact with the second contacted particle as opposed to being overlapped.
    """

    dropped_particle_size, dropped_particle_location = particle_info[0], particle_info[1]
    first_contacted_size, first_contacted_location = particle_info[2], particle_info[3]
    second_contacted_size, second_contacted_location = particle_info[4], particle_info[5]
    smallest_particle_size = particle_info[6]

    # Set delta and epsilon
    epsilon = .001
    delta = -.1
    if .001 < (1/5) * smallest_particle_size:
        epsilon = .001
    else:
        epsilon = (1/5) * smallest_particle_size
    going_up = True
    # Rotate and check particle until sufficiently close to in contact with second contacted particle
    start = time.time()
    while abs(math.dist(dropped_particle_location, second_contacted_location) - (dropped_particle_size + second_contacted_size)) > epsilon:
        # Rotate particle
        normalized = [i - j for (i,j) in zip(dropped_particle_location, first_contacted_location)]
        incremented = increment_spherical_angle(normalized, math.radians(delta))
        rotated_coordinates = incremented[:-1]
        dropped_particle_location = [i + j for (i,j) in zip(rotated_coordinates, first_contacted_location)]

        # Check if it should be moved up or down. If there is a switch, half delta_z and switch direction
        if check_overlap((dropped_particle_size, dropped_particle_location), (second_contacted_size, second_contacted_location)):
            if going_up == True:
                pass
            else:
                going_up = True
                delta *= (-1/2)
        else:
            if going_up == False:
                pass
            else:
                going_up = False
                delta *= (-1/2)


    return dropped_particle_location
 
def adjust_third_contact_location(particle_info, triangle_vertices, smallest_particle_size):
    """
    Given particle info, triangle vertices, and smallest_particle_size, adjusts location of dropped particle 
    so in contact with third contacted particle   
    """

    dropped_particle_index, dropped_particle_size, dropped_particle_location = particle_info[0], particle_info[1], particle_info[2]
    third_contacted_size, third_contacted_location = particle_info[3], particle_info[4]

    # Set delta and epsilon
    epsilon = .001
    delta = -.1
    if .001 < (1/5) * smallest_particle_size:
        epsilon = .001
    else:
        epsilon = (1/5) * smallest_particle_size
    going_up = False

    # Rotate and check particle until sufficiently close to in contact with second contacted particle
    while abs(math.dist(dropped_particle_location, third_contacted_location) - (dropped_particle_size + third_contacted_size)) > epsilon:
        print(check_overlap((dropped_particle_size, dropped_particle_location), (third_contacted_size, third_contacted_location)))
        # Rotate particle

        dropped_particle_location = rotate_triangle(triangle_vertices, dropped_particle_index, delta)[dropped_particle_index]                                                                     
        # Check if it should be moved up or down. If there is a switch, half delta_z and switch direction
        if check_overlap((dropped_particle_size, dropped_particle_location), (third_contacted_size, third_contacted_location)):
            if going_up == True:
                pass
            else:
                going_up = True
                delta *= (-1/2)
        else:
            if going_up == False:
                pass
            else:
                going_up = False
                delta *= (-1/2)
        

    return dropped_particle_location

def narrow_first_contact(overlapping, dropped_particle_size, temporary_location):
    """
    Given a list of overlapping particles, where each element if a tuple of a vertex index, a size and a location,
    a dropped particle size and a temporary dropped particle location, returns the index of particle that was first contacted 
    by the dropped particle during the first motion
    """
    modifiable_overlapping = copy.deepcopy(overlapping)
    delta_z = 100
    going_up = True

    while len(modifiable_overlapping) != 1:
    
        # Change z-coordinate and check for overlap
        temporary_location[2] += delta_z
        new_overlapped = []
        for overlapped_particle in overlapping:
            if check_overlap((dropped_particle_size, temporary_location), (overlapped_particle[1], overlapped_particle[2])):
                if overlapped_particle not in modifiable_overlapping:
                    modifiable_overlapping.append(overlapped_particle)
            else:
                if overlapped_particle in modifiable_overlapping:
                    modifiable_overlapping.remove(overlapped_particle)
        num_overlapped = len(modifiable_overlapping)

        # Increment downward if went too far up
        if num_overlapped == 0:
            if going_up == False:
                pass
            else:
                delta_z *= -(1/2)
                going_up = False
        
        # Increment downward if went too far down
        elif num_overlapped > 1:
            if going_up == True:
                pass
            else:
                delta_z *= -(1/2)
                going_up = True

    return modifiable_overlapping[0][0]

def narrow_second_contact(soil_graph, overlapping, dropped_particle_size, dropped_particle_location, first_contacted_index):
    """
    Given a soil graph, a list of particle indices currently being overlapped, a dropped particle index, and the index of the 
    first contacted particle, returns the first particle that was contacted on the rotational movement
    """
    modifiable_overlapping = copy.deepcopy(overlapping)
    delta = -3
    going_up = True
    first_contacted_size, first_contacted_location = soil_graph.vertices[first_contacted_index]['size'], soil_graph.vertices[first_contacted_index]['coordinates']

    while len(modifiable_overlapping) != 1:

        # Rotate and check overlap
        normalized = [i - j for (i,j) in zip(dropped_particle_location, first_contacted_location)]
        incremented = increment_spherical_angle(normalized, math.radians(delta))
        rotated_coordinates = incremented[:-1]
        dropped_particle_location = [i + j for (i,j) in zip(rotated_coordinates, first_contacted_location)]
        new_overlapped = []
        for overlapped_particle in overlapping:
            overlapped_size, overlapped_location = soil_graph.vertices[overlapped_particle]['size'], soil_graph.vertices[overlapped_particle]['coordinates']
            if check_overlap((dropped_particle_size, dropped_particle_location), (overlapped_size, overlapped_location)):
                if overlapped_particle not in modifiable_overlapping:
                    modifiable_overlapping.append(overlapped_particle)
            else:
                if overlapped_particle in modifiable_overlapping:
                    modifiable_overlapping.remove(overlapped_particle)
        num_overlapped = len(modifiable_overlapping)

        # Increment downward if went too far up
        if num_overlapped == 0:
            if going_up == False:
                pass
            else:
                delta *= -(1/2)
                going_up = False
        
        # Increment downward if went too far down
        elif num_overlapped > 1:
            if going_up == True:
                pass
            else:
                delta *= -(1/2)
                going_up = True
    
    return modifiable_overlapping[0]

def narrow_third_contact(soil_graph, overlapping, dropped_particle_index, dropped_particle_size, triangle_vertices, delta):
    """
    Given a soil graph, overlapping particles,  a dropped particle index, a dropped particle size, 
    and a triangle to rotate returns the first particle that was contacted on the  third rotational 
    movement
    """

    # Define dynamic variables
    modifiable_overlapping = copy.deepcopy(overlapping)
    delta *= -1
    going_up = True
    
    while len(modifiable_overlapping) != 1:

        # Rotate and check overlap
        dropped_particle_location = rotate_triangle(triangle_vertices, dropped_particle_index, delta)[dropped_particle_index]

        new_overlapped = []
        for overlapped_particle in overlapping:
            overlapped_size, overlapped_location = soil_graph.vertices[overlapped_particle]['size'], soil_graph.vertices[overlapped_particle]['coordinates']
            if check_overlap((dropped_particle_size, dropped_particle_location), (overlapped_size, overlapped_location)):
                if overlapped_particle not in modifiable_overlapping:
                    modifiable_overlapping.append(overlapped_particle)
            else:
                if overlapped_particle in modifiable_overlapping:
                    modifiable_overlapping.remove(overlapped_particle)
        num_overlapped = len(modifiable_overlapping)

        # Increment downward if went too far up
        if num_overlapped == 0:
            if going_up == False:
                pass
            else:
                delta *= -(1/2)
                going_up = False
        
        # Increment downward if went too far down
        elif num_overlapped > 1:
            if going_up == True:
                pass
            else:
                delta *= -(1/2)
                going_up = True
    
    return modifiable_overlapping[0]
    

####################################
# FUNCTIONS TO FIND CONTACT POINTS #
####################################

def find_first_contact(soil_graph, dropped_particle_size, location):
    """
    Given a list of relevant_particles, where each element of the list is a tuple of a particle index and z-coordinate,
    a soil graph of the form returned by create_soil_graph, an xy location, and a radius, returns a tuple with the index of
    the first particle that the dropped particle comes into contact with and the final location of the particle
    """
    # Check for relevant particles
    relevant_particles = find_first_relevant_particles(soil_graph, dropped_particle_size, location)
    if not relevant_particles:
        return None
    
    # Define variables
    z_to_check = []
    temporary_location = None
    first_contact = None
    particle_dictionary = soil_graph.vertices
    
    # Initialize z-coordinates to check
    first_particle = relevant_particles[0]
    highest_z = first_particle[1]
    center_z = particle_dictionary[first_particle[0]]['coordinates'][2]
    equal_centers = center_z - dropped_particle_size
    bisect.insort(z_to_check, highest_z, key=lambda x: -1 * x)
    bisect.insort(z_to_check, center_z, key=lambda x: -1 * x)
    bisect.insort(z_to_check, equal_centers, key=lambda x: -1 * x)
    particle_splice_index = 0

    # If there are more z-coordinates to check, keep going
    while z_to_check:

        # Temporarily place particle at highest z-coordinate and check overlap
        temporary_location = [location[0], location[1], z_to_check[0] + dropped_particle_size]
        del z_to_check[0]
        overlapping = []
        for particle in relevant_particles[:particle_splice_index + 1]:
            particle_size = particle_dictionary[particle[0]]['size']
            particle_location = particle_dictionary[particle[0]]['coordinates']
            if check_overlap((dropped_particle_size, temporary_location), (particle_size, particle_location)):
                overlapping.append((particle[0], particle_size, particle_location))

        # If overlapping particle found, narrow down to first contacted and break loop
        if overlapping:
            if len(overlapping) > 1:
                first_contact = narrow_first_contact(overlapping, dropped_particle_size, temporary_location)
            else:
                first_contact = overlapping[0][0]
            break
        
        # If no overlap was found, move on to next particle
        if particle_splice_index == len(relevant_particles) - 1:
            continue
        else:
            particle_splice_index += 1
            next_particle = relevant_particles[particle_splice_index]
            top_of_sphere = particle[1]
            center_of_sphere = particle_dictionary[particle[0]]['coordinates'][2]
            equal_centers = center_of_sphere - dropped_particle_size
            bisect.insort(z_to_check, top_of_sphere, key=lambda x: -1 * x)
            bisect.insort(z_to_check, center_of_sphere, key=lambda x: -1 * x)
            bisect.insort(z_to_check, equal_centers, key=lambda x: -1 * x)
    

    # ADJUST LOCATION SO IT RESTS ON CONTACTED PARTICLE
    if first_contact != None:
        first_contact_size, first_contact_location = particle_dictionary[first_contact]['size'], particle_dictionary[first_contact]['coordinates']
        final_location = adjusted_location(soil_graph, (dropped_particle_size, temporary_location), (first_contact_size, first_contact_location))
        return first_contact, final_location, first_contact_size, first_contact_location
    else:
        return None

def find_second_contact(soil_graph, particle_info):
    """
    Given a soil graph, and a tuple containing dropped particle index, dropped particle size, dropped particle current location,
    first contact index, first_contact_size, and first_contact location, finds the second contacted particle based of rotational
    movement.
    """
    # Define static variables
    dropped_particle_index, dropped_particle_size, dropped_particle_location = particle_info[0], particle_info[1], particle_info[2]
    first_contacted_index, first_contacted_size, first_contacted_location = particle_info[3], particle_info[4], particle_info[5]
    relevant_particles = surrounding_particles(soil_graph, dropped_particle_size, first_contacted_size, first_contacted_location)
    relevant_particles.remove(first_contacted_index)

    # Rotate particle and iterate through relevant until an overlapping particle is found
    overlapping = []
    while not overlapping:

        normalized = [i - j for (i,j) in zip(dropped_particle_location, first_contacted_location)]
        incremented = increment_spherical_angle(normalized, math.radians(5))
        rotated_coordinates = incremented[:-1]
        dropped_particle_location = [i + j for (i,j) in zip(rotated_coordinates, first_contacted_location)]
        at_bottom = incremented[-1]
        for particle in relevant_particles:
            if check_overlap((soil_graph.vertices[particle]['size'], soil_graph.vertices[particle]['coordinates']), (dropped_particle_size, dropped_particle_location)):
                overlapping.append(particle)

        # Check if at bottom of particle
        if at_bottom:
            break
        
        # Check if hit ground
        if dropped_particle_location[2] <= 0:
            break
    # Narrow down to only one overlapping particle if more than one more contacted at once
    if len(overlapping) > 1:
        second_contact_index = narrow_second_contact(soil_graph, overlapping, dropped_particle_size, dropped_particle_location, first_contacted_index)
    elif len(overlapping) == 1:
        second_contact_index = overlapping[0]
    else:
        second_contact_index = None

    # Adjust for overlap
    if second_contact_index:
        second_contact_info = soil_graph.vertices[second_contact_index]['size'], soil_graph.vertices[second_contact_index]['coordinates']
        new_particle_info = [dropped_particle_size, dropped_particle_location, first_contacted_size, first_contacted_location, second_contact_info[0], second_contact_info[1], soil_graph.smallest_particle]
        dropped_particle_location = adjust_second_contact_location(soil_graph, new_particle_info, dropped_particle_index)

    return second_contact_index, dropped_particle_location, relevant_particles

def find_third_contact(soil_graph, particle_info, relevant_particles):
    """
    Given a soil graph, and a tuple containing dropped particle index, dropped particle size, dropped particle current location,
    first contact index, first_contact_size, first_contact location, second contact index, second_contact_size, and second_contact_location,
    and finally a list of relevant particles, finds the third contacted particle based of rotational movement
    """
    # Define static variables
    dropped_particle_index, dropped_particle_size, dropped_particle_location = particle_info[0], particle_info[1], particle_info[2]
    first_contacted_index, first_contacted_size, first_contacted_location = particle_info[3], particle_info[4], particle_info[5]
    second_contacted_index, second_contacted_size, second_contacted_location = particle_info[6], particle_info[7], particle_info[8]
    if second_contacted_index in relevant_particles:
        relevant_particles.remove(second_contacted_index)

    # Define dynamic variables
    check = True
    at_bottom = False
    delta = math.radians(1)
    total_delta = 0
    lowest_z = copy.deepcopy(dropped_particle_location[2])

    # Find first particle that is contacted upon rotating against the two contacted particles
    overlapping = []
    while not overlapping:
        # Define object to be rotated
        triangle_vertices = {
            dropped_particle_index: dropped_particle_location, 
            first_contacted_index: first_contacted_location,
            second_contacted_index: second_contacted_location
            }

        # Check which direction to be rotating
        if check == True:
            rotated_location = rotate_triangle(triangle_vertices, dropped_particle_index, delta)[dropped_particle_index]
            if dropped_particle_location[2] < rotated_location[2]:
                delta *= -1
            check = False

        # Rotate particle and update dynamic variables
        dropped_particle_location = rotate_triangle(triangle_vertices, dropped_particle_index, delta)[dropped_particle_index]
        if lowest_z > dropped_particle_location[2]:
            lowest_z = copy.deepcopy(dropped_particle_location[2])

        # Check if overlapping any particles
        for particle in relevant_particles:
            if check_overlap((soil_graph.vertices[particle]['size'], soil_graph.vertices[particle]['coordinates']), (dropped_particle_size, dropped_particle_location)):
                overlapping.append(particle)

        # If rotated below z-axis, break
        if dropped_particle_location[2] <= 0:
            break

        # If one full revolution has been made, break when the particle gets back to its lowest_z:
        if abs(total_delta) > 2*math.pi:
            if dropped_particle_location[2] <= math.ceil(lowest_z):
                break
        total_delta += delta

    # Narrow down to the first contact
    if len(overlapping) > 1:
        third_contact_index = overlapping[0]
    else:
        third_contact_index = None


    return third_contact_index, dropped_particle_location

############################
# FUNCTIONS FOR SIMULATION #
############################

def drop_particle(soil_graph, dropped_particle_index, location):
    """
    Given a soil graph of the form returned from create_soil_graph, a vertex index, and an (x,y) tuple
    containing xy-coordinates of where it will be dropped. If the particle does not come into contact with any other 
    particles, rests at z = 0. 
    """
    # Define static variable
    particle_dictionary = soil_graph.vertices
    dropped_particle_size = particle_dictionary[dropped_particle_index]['size']

    # LOGIC FOR MOTION OF DROPPED PARTICLE 
    first_contact = (None, None)
    temporary_location = [location[0], location[1], 0]
    # If a first particle is contacted, begin to move rotationally
    first_contact = find_first_contact(soil_graph, dropped_particle_size, location)
    if first_contact != None:
        soil_graph.add_edge((dropped_particle_index, first_contact[0]))
        temporary_location, first_contact_size, first_contact_location = first_contact[1], first_contact[2], first_contact[3]
        
        # Check if particle makes second contact upon rotating
        particle_info = [dropped_particle_index, dropped_particle_size, temporary_location, first_contact[0], first_contact_size, first_contact_location]
        second_contact = find_second_contact(soil_graph, particle_info)
        second_contact_index, temporary_location, relevant_particles = second_contact[0], second_contact[1], second_contact[2]

        # If a second particle is contacted, move rotationally based off those two contacted particles
        if second_contact_index != None:
            soil_graph.add_edge((dropped_particle_index, second_contact_index))
            second_contact_info = particle_dictionary[second_contact_index]
            particle_info += [second_contact_index, second_contact_info['size'], second_contact_info['coordinates']]
            particle_info[2] = temporary_location
            third_contact = find_third_contact(soil_graph, particle_info, relevant_particles)
            third_contact_index, temporary_location = third_contact[0], third_contact[1]

            # If third particle is contacted, add as neighbor and complete movement
            if third_contact_index != None:
                 soil_graph.add_edge((dropped_particle_index, third_contact_index))

    final_location = temporary_location
    soil_graph.assign_coordinate(dropped_particle_index, final_location)


def particle_simulation(soil_graph):
    """
    Given a soil graph of the form returned by create_soil_graph, rearrange particle coordinates
    to simulate an actual soil
    """
    dimension = soil_graph.boundary_size
    particle_dictionary = soil_graph.vertices
    total = 0
    # Drop each particle at a random (x,y) location
    for particle in particle_dictionary.items():
        if total > 100:
            return False
        start = time.time()
        drop_location = (random.uniform(0,dimension), random.uniform(0,dimension))
        drop_particle(soil_graph, particle[0], drop_location)
        end = time.time()
        total += end - start
    return True

def visualize_graph(soil_graph):
    """
    Given a soil graph of the form returned from create_soil_graph, displays a 3D scatter plot of particles,
    where each point represents the center of a particle.
    """
    # Particle information
    list_centers = []
    list_radii = []
    list_indices = []
    for ix, particle in soil_graph.vertices.items():
        
        if particle['coordinates'] == None:
            continue
        list_centers.append(particle['coordinates'])
        list_radii.append(particle['size'])
        list_indices.append(ix)

    # Draw spheres and plot figures
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    counter = 0
    for c, r, index in zip(list_centers, list_radii, list_indices):
        counter += 1
        colors = ['g','b', 'r', 'y']
        current_color = colors[counter % 4 - 1]
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = r*np.cos(u)*np.sin(v)
        y = r*np.sin(u)*np.sin(v)
        z = r*np.cos(v)
        ax.plot_surface(x + c[0], y + c[1], z + c[2], color= current_color, alpha=0.7)
        ax.text(c[0], c[1], c[2], index, color=(0, 0, 0, 1), fontsize=10, ha='center', va='center') 
    ax.set_box_aspect([1,1,1])
    plt.show()

if __name__ == '__main__':

    # FULL SAMPLES

    # (40.1 percent clay, 12.0 percent silt, 48.0 percent sand)
    test_sample_1 = [[491.63087133531207, 73], [0.009442950719121634, 31], [0.26102925749697165, 20], [60.135295998785956, 3], [76.93398000103714, 5], [0.17567028364555906, 27], [0.18330064990762227, 27], [8.168732756797475, 9], [351.4149290174114, 90], [9.053903158447065, 8], [0.5290659969267004, 1], [0.44061119470840626, 2], [0.26264070639388304, 19], [0.183909632480343, 27], [558.3958408575741, 57], [0.4334638324653902, 3], [3.727414240453601, 10], [674.9828471746202, 36], [28.92361548735273, 5], [0.44541803980129363, 2], [62.253743092469975, 3], [1.121772039121354, 1], [4.136696800867748, 11], [434.86179680500663, 85], [0.05079935447915931, 5], [55.21752105604084, 3], [0.16583607327732988, 27], [47.4075033932498, 2], [0.12319363687949285, 22], [4.488882575272337, 12], [1.2241216846677732, 1], [0.10814039506689231, 19], [0.1283822060412815, 23], [23.01653181293176, 6], [28.824154836787642, 5], [0.2360012426773636, 23], [252.7762191212329, 66], [0.05268752426985628, 5], [0.1268446264888266, 22], [1.0760452416302608, 1], [35.711817119681, 3], [0.029151596573523814, 16], [1070.7874379657217, 5], [1155.287443730849, 3], [0.060585196500637706, 7], [0.41090671942998824, 3], [0.016836980354940873, 44], [1.4584400572056957, 1], [0.013407296501367877, 64], [0.04184423530326855, 4], [5.20563241662135, 13], [0.010455858654795225, 44], [2.3958815447412762, 5], [0.05609131248817404, 6], [1135.3423441829189, 4], [6.235563437815565, 12], [59.36708322735155, 3], [0.014561908704803717, 59], [0.007420843005344983, 11], [0.008464622522327877, 20], [9.067199787345439, 8], [3.787660902462398, 10], [287.92510863508016, 79], [0.29632197656971065, 14], [345.3786914496712, 90], [0.08664881222907933, 13], [414.16258822587935, 88], [11.09797932202522, 6], [2.1295359494261503, 4], [4.404684796065286, 12], [912.5061822476524, 13], [7.71941510145328, 10], [4.372091959744154, 12], [221.5024336534207, 54], [511.4340968615268, 68], [36.9184967764338, 3], [2.87543882655541, 7], [889.9137208503901, 15], [0.02050796571959193, 32], [0.3559705188225168, 7], [5.720700379857568, 13], [213.67420810396771, 50], [0.030378110863647518, 14], [0.2526671681694055, 21], [3.1630662307644584, 8], [0.3873872788396682, 5], [154.63551192641197, 26], [0.07676131076416683, 11], [0.2996534507747318, 14], [22.981185910113755, 6], [1.55127375825411, 2], [58.0918161131908, 3], [0.05169934222518969, 5], [12.149847388518564, 6], [85.52858828337405, 7], [0.29210945991830667, 15], [8.211043664955657, 9], [6.509092902223802, 12], [0.02549418757056616, 23], [1038.933722286336, 7], [149.7698358572607, 25], [0.07672400279475382, 11], [0.005370777472182133, 2], [5.41812395921388, 13], [0.18869472517729455, 27]]
    
    # (17.6 percent clay, 45.5 percent silt, 36.9 percent sand)
    test_sample_2 = [[0.21869010471634934, 6], [0.018841797820426418, 21], [0.02509094765244421, 21], [2.26015475288893, 4], [247.2574286917464, 66], [1.4426381816218303, 1], [0.03250072475105271, 24], [30.27502903512779, 53], [174.14313253405038, 55], [0.06906054301877869, 13], [6.525423656200814, 55], [0.008805614528009037, 1], [1.5977140855548027, 1], [0.10180104759556556, 13], [459.7861316040727, 6], [0.00976261340479878, 1], [0.009936214955039201, 2], [1.6497734059083602, 1], [0.17739549226056237, 9], [11.173747163196232, 40], [0.029495222103728954, 22], [0.05170067085119651, 17], [0.008433663280059539, 1], [511.6218276312647, 2], [177.1260991426652, 57], [0.1047262831150693, 13], [221.9763165468313, 69], [130.1685935957951, 32], [172.6405325698826, 55], [0.02601936202548221, 21], [0.011465118137486277, 3], [0.02462749243803282, 21], [0.16676867721049257, 10], [0.2436994966404156, 4], [36.14184687320243, 38], [2.87325532025443, 8], [44.765315575971684, 16], [0.3301665910790528, 1], [5.350355743152077, 40], [313.9246326176714, 40], [9.272844717820465, 55], [2.123159188640916, 3], [281.2450694325433, 54], [0.15960030000274322, 10], [51.442138474331195, 8], [0.10191911627822056, 13], [246.34078693404618, 66], [5.034943123059267, 35], [25.09141753696531, 57], [135.3251068254787, 34], [315.230854681065, 40], [0.028405605508210054, 21], [42.62863297423857, 20], [403.68533691349774, 14], [2.400930720442561, 5], [15.800678879747334, 38], [0.2071152579669397, 7], [2.860323413268527, 8], [0.03638467668487883, 23], [31.492395203890197, 50], [105.0014177171851, 19], [0.32492456861872, 1], [11.88253223256762, 36], [1.2537054070388112, 1], [1.625076521690445, 1], [13.989964764248285, 34], [1.5724104206872784, 1], [0.011366369012471055, 3], [0.1636961250913646, 10], [0.011296948247342915, 3], [0.015905509577314236, 14], [2.875588116621202, 8], [2.705498720086275, 7], [3.9709480665040693, 20], [8.781569438777652, 58], [15.68862803927134, 38], [96.15527701564336, 15], [16.97228053468291, 42], [0.012616593964717803, 5], [393.9411094867155, 16], [2.1851133062799373, 3], [0.14276830853863107, 12], [36.24938600295522, 38], [31.657953217875097, 50], [434.11646715142894, 9], [1.4197022401634247, 1], [366.76692647001425, 23], [6.303404202922883, 53], [182.5192069839734, 59]]
    
    # (19.1 percent clay, 76.8 percent silt, 4.1percent sand)
    test_sample_3 = [[1.04480567122153, 1], [7.011325156194995, 65], [0.026933159651266567, 16], [0.04293699399390451, 20], [28.504478674505027, 60], [0.03760176431675787, 18], [2.5416247284488853, 7], [24.40211523933054, 69], [3.354059808992372, 14], [243.09047749744377, 10], [293.00097983554355, 9], [2.0984665427260314, 5], [4.096832479596256, 22], [0.07384422742210146, 7], [13.74618523435094, 90], [0.05073342301986569, 15], [12.396896974972407, 96], [0.013360783424969876, 24], [6.495872510112946, 57], [0.038634607903312, 19], [183.4932397791009, 14], [0.017338238985053114, 39], [0.02270452516709483, 29], [11.216405980260452, 98], [18.7634800136725, 75], [9.187524048914176, 91], [1.1690899013407914, 1], [2.0719516849256987, 4], [885.4065899226824, 1], [103.6533879103388, 9], [13.102598131363065, 93], [653.3942146600372, 5], [0.02057203553512573, 37], [0.030467975795961052, 14], [3.4388083956164683, 15], [12.218877426462553, 96], [1.929661945480064, 4], [23.312631912390643, 70], [828.5130208851577, 2], [448.9924099814581, 7], [35.64476068276021, 34], [44.641351040712685, 10], [353.6225626741712, 8], [11.689042363076052, 98], [0.11005958503772431, 4], [0.01576307167577434, 34], [36.610581326674016, 30], [0.04536454252926868, 19], [0.0339813501392829, 16], [2.5935353407475605, 8], [8.59513287393456, 86], [26.168222859502755, 66], [51.65660756099795, 4], [31.35965330741733, 50], [0.11063562207366111, 4], [0.00561138980531077, 1], [196.96832118754546, 13], [10.893852771506163, 98], [0.022950448882676736, 28], [0.049595044325959695, 16], [0.03394415713858214, 16], [4.7052401180110355, 30]]

    # (64.7 percent clay, 27.8 percent silt, 7.5 percent sand)
    test_sample_4 = [[492.7601289385319, 2], [0.029758967983811016, 33], [0.38550798865381414, 4], [0.0755299097892722, 70], [11.762687316463769, 18], [1.0931130484471872, 1], [0.02974873663370547, 33], [14.726031302559052, 17], [9.030667860206293, 20], [0.008929726468384398, 19], [0.07213411267629678, 69], [0.0454659869306705, 53], [4.171129036605869, 13], [0.03222140118020873, 36], [2.7827553646964343, 7], [13.45727630647046, 18], [8.578569324615575, 20], [0.2080533392631975, 29], [504.8182950216846, 2], [0.02564184052799465, 32], [0.13781842616939108, 49], [4.041637181076005, 12], [6.020334360347126, 19], [1.5539683040807266, 2], [0.06112633158351893, 66], [24.16982030083122, 16], [0.009405728110727561, 21], [617.7560757934162, 1], [67.78633768476034, 7], [25.525565420969937, 15], [6.17279890678791, 19], [32.504867158778524, 13], [7.876460131069038, 20], [27.363464356926475, 15], [0.03624315738504083, 41], [0.0620587137231103, 66], [38.95673057105803, 11], [0.004578809338353996, 2], [3.3305757449336277, 9], [218.67346365326233, 11], [643.9837218074978, 1], [13.27137207978768, 18], [0.0849520318184554, 69], [4.318875440223291, 13], [0.004659908212395379, 2], [0.004376342239596883, 1], [1.8006583674499852, 3], [93.39236528888001, 11], [32.40470498693219, 13], [6.826276757857931, 20], [0.0234231609143625, 33], [14.697423689328184, 18], [0.8615106615670537, 1], [334.39846490722783, 5], [469.0972818857193, 2], [0.8192620355056139, 1], [0.15712975387895378, 42], [0.18395130104027968, 34], [837.0011363279785, 1], [160.8629592771917, 15], [0.446101126818213, 1], [282.98158630839646, 7], [1.2649917904563546, 1], [0.018955104008972396, 41], [124.95710060414457, 15], [141.5946946414488, 16], [99.49907058725006, 12], [4.902893989943604, 16], [437.9235115653347, 3], [288.0382024067276, 7], [5.934333586677983, 18], [455.92873639748734, 3], [378.7268451219504, 4], [0.24789645162572016, 21], [27.223767822412352, 15], [18.270660350235165, 17], [0.25279276337647594, 20], [0.046271643701478314, 54], [724.1673147760071, 1], [190.80234568285942, 13], [5.257387783431824, 17], [2.147233961625421, 4], [110.82088991742856, 14], [0.14554406526319946, 46], [0.16915089660172503, 38], [7.055630073166493, 20], [608.0457807987672, 1], [32.289250582491476, 13], [48.6879809459267, 8], [0.7988450727915422, 1], [25.4317000077947, 15], [0.45465896126313493, 1], [0.00811222332377951, 14], [12.24600948342963, 18], [48.32652744157965, 8], [0.3202785798920488, 9], [0.04549796071637024, 53], [6.248397493909996, 19], [3.0343712368751263, 8], [0.2711304876741101, 17], [2.770351213292009, 6], [0.003595342648647854, 1], [0.12086989263009253, 56], [48.140306268474326, 8], [0.062442387575100636, 66], [0.007049546108161387, 9], [11.114143231636044, 18], [0.07682551835087531, 70]]
    
    # (25.9 percent clay, 40.4 percent silt, 33.7 percent sand)
    test_sample_5 = [[0.01839682020921751, 7], [1.729365513717588, 2], [186.30955291971367, 47], [235.4825198834645, 33], [18.66054457383379, 23], [5.608027921398218, 39], [0.08436519896257436, 41], [1.6403358070668925, 2], [320.4815717486666, 35], [0.02688827531497461, 15], [20.952404284788102, 18], [6.7345945191984695, 49], [2.8660926100177986, 8], [380.1966107549469, 32], [363.51815201932544, 34], [0.16263899822036432, 1], [0.13360428431166893, 5], [481.1101425590125, 21], [0.00625846821640891, 1], [5.48229172473354, 38], [16.910725944556955, 27], [124.75479630611015, 46], [0.028389295787446974, 17], [0.1594521976840935, 1], [0.09316527928820145, 31], [0.17345390041634137, 1], [62.14002206091643, 8], [0.031149296140117754, 22], [18.796583826062434, 23], [45.436878212278074, 4], [1.3314636635715813, 1], [0.05094875301310263, 50], [4.178292801196353, 21], [33.5616821384417, 5], [33.674650917658035, 5], [0.011077358782331022, 26], [5.081847891321726, 33], [4.708298182489889, 28], [8.171562948490381, 53], [0.018635808556346452, 6], [0.049137137643038474, 48], [0.1258399164552187, 7], [8.731031774214868, 53], [0.09358617145966813, 30], [24.431318404195533, 12], [65.00961085188422, 9], [86.00435923075824, 20], [14.029336327548092, 36], [1.2690443019743798, 1], [652.4373308078767, 5], [0.030870682851226742, 21], [1.1059170987462967, 1], [25.177328113639067, 11], [5.824282598849694, 41], [4.642874370089781, 27], [0.013343543064396111, 35], [5.684395484253921, 40], [7.898544759779514, 53], [14.085868744137638, 36], [2.0036459889912166, 3], [287.9175656109578, 35], [60.90360631266693, 8], [522.2223892240415, 16], [0.010253342831126317, 19], [136.1615525298335, 52], [42.040850601745824, 4], [24.500293361412243, 12], [160.29135530373563, 55], [2.6340530277164413, 7], [9.476757940206914, 51], [0.05993725506541296, 56], [2.8761913139251734, 8], [0.014930188693590439, 25], [274.8250413900655, 34], [573.4226852152993, 11], [262.683714576127, 33], [14.655354554952606, 34], [1.9795970350163061, 3], [2.434590644955596, 5], [37.701709148864985, 4], [2.3984695655505077, 5], [129.96205954171396, 49], [0.12789779328731837, 6], [0.12102223447946746, 9], [0.09228766130129822, 32], [119.44611682724363, 43], [160.08455871778762, 55]]   

    
    # Particle Drop Samples
    drop_sample_1 = [[235.4825198834645, 1], [76.93398000103714, 1], [14.726031302559052, 1]]

    drop_sample_2 = [[492.7601289385319, 4], [247.2574286917464, 4], [800, 1], [1200, 1], [100, 1], [50, 1]]

    drop_sample_3 = [[15,5],[10,5],[5,5], [10,5]]

    ######## TESTING ###########

    soil_graph_1 = create_soil_graph(test_sample_1)
    start = time.time()
    particle_simulation(soil_graph_1)
    end = time.time()
    print(end - start)

"""
#########
# Notes #
#########

Plan:
    1) Assign coordinates to particles in a sensible way

        - What is a sensible way to assign coordinates to particles?
            i) One idea is to just put them down randomly, which is how I am going to begin
            ii) Another way could maybe be to try emulating how the particles might roughly arrange 
            themselves according to size or initial place in the graph.

        A: For now, I am just going to randomly put the particles in and carry out from there. I can go back and change this.
        The approach to how particles are connected shouldn't depend upon this

        - How should coordinates be assigned so that no particles interfere with each other.

        A: Just make all be contained in a ball of radius 2 (cm)


    2) Change coordinates of particles in a sensible way to make it more realistic of a soil

        ALGORITHM:

        (1) Choose a particle at random and assign it coordinates
        (2) Select an "appropriate" neighbor for that particle
        (3) Connect the neighbor


PLAN FOR TOMORROW(1/24/23):

- Test to make sure that I am accurately finding relevant particles
    - Make sure units are correct

    (1) Create a few samples (Done)
    (2) Create a visualizer (Done)

Now, the first order of action is to drop particles at a random location, seeing if it comes into contact with any particles on the way down

The only information one needs to decide the boundary size and container size is the size of the particles.

PLAN FOR TOMORROW (1/25/23):

- Debugging
    - First, debugging relevant_containers. x,y bound variables are not always being defined

PLAN FOR TOMORROW (1/27/23)
- Begin debugging first contact code. Run ample tests to make sure it is working.
   The problem in this area is that I should be iterating through z-levels instead of particles or
   something like this. For instance, if there is only one relevant particle, the way the code is written right now
   it just checks the top level of that particle and that is it. I need it to go to every z-level not every particle.
   I should just have a counter that moves along the relevant particles while I iterate through the z-levels.

- Move on to secondary movement.

PLAN FOR NEXT DAY (1/30/23)
- Fix adjusted_location function. Running really slow but could be running crazy quickly.

Documentation Purposes:

1/24/23

 - Decided to try a more memory-intensive approach than time-intensive. First step in this direction was partitioning the box that the particles are 
 going to be dropped to find relevant particles quicker.

 - At this point in my code, I should be able to drop particles in a box randomly and assign coordinates and
place them into containers based on their coordinates. The particles do not interact with each other yet, so upon running, 
I expect them to overlap and all be at z = 0.

1/25/23

- For now, when a particle is dropped, it queries every particle in the relevant containers. Idea to increase time efficiency: organize 
containers into descending z-order. 

- Way of organizing vertices is done, need to check to make sure it's working correctly though

1/26/23

- First order of business is making sure that the grid system is working properly and when a particle is dropped, relevant particles are being given.
- To do this, it seems necessary to make a visualizer of spheres that are scaled properly
- Ideas for managing space complexity. Use array instead of lists for better memory allocation.

2/2/23

- Linear motion is working and rotational motion is in the works.
- The basic procedure for implementing the rotational motion is this:
    1. Based on how the particle rests, determine a rotational movement
    2. Based on the movement, determine the relevant particles
    3. Move according to some rotational movement, checking overlap with the relevant particles
    4. Maybe move in intervals of the smallest relevant particle around it

1: Determining a rotational movement
- When the particle is only in contact with one particle,
  the motion is only to decrease phi. Theta stays the same. In all motions, 
  the radius stays the same. Now, I just need to figure out how to move the
  particle based on how it rests on two particles. For now, I think I will just
  write code to move it that takes in two parameters.


FOR NEXT TIME I COME BACK (2/13):

Simulation software is working for the most part. Slight bug on adjusting the overlapping
of the first rotational movement, it is just when the particle rotates to the z-axis the function
gets weird for some reason. When running the simulation only bugs occasionally do to this, so not
going to fix right now.

Now I am implementing the third contact motion. The first function I need to write and really make sure it works
is rotating the particle. Given all of the particle information of sizes and locations, rotate the particle according to
some theta radians. The rest of the simulation should be pretty straight-forward if I can get this working right.


(2/19)
PLAN FOR TODAY:
Hard code an example of three particle system and develop a rotation function that works robustly.

(2/26/23)
Particle simulation is done for the most part. Now beginning to write analytics.py. This past week consisted of
debugging what I had
"""