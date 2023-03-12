import math
import copy
import bisect

#################
# ERROR CLASSES #
#################

class VertexNotPresentError(Exception):
    pass

class ImproperEdgeError(Exception):
    pass

class PositionNotDefinedError(Exception):
    pass

####################
# HELPER FUNCTIONS #
####################
def find_relevant_containers(soil_graph, particle_size, location):
    """
    Given a soil graph of the form returned by create_soil_graph, a particle_size, and an xy-location, returns
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
            new = (round(lower_x + i * partition_length, 2), round(lower_y + j * partition_length, 2))
            containers.append(new)
    return containers

def bisect_search(vertex_list, particle_z):
    """
    Given a list of tuples and a particle z, find index to insert particle
    such that the list stays in non-increasing order.
    """
    if len(vertex_list) == 0:
        return 0
    
    # Particle is higher than any in that container
    if particle_z > vertex_list[0][1]:
        return 0
    
    # Particle is lower than any in that container
    if vertex_list[-1][1] > particle_z:
        return len(vertex_list)
    
    # Bisection search if none of the above
    lo, hi = 0, len(vertex_list) - 1
    while lo < hi:

        if particle_z == vertex_list[lo][1]:
            return lo
        elif particle_z == vertex_list[hi][1]:
            return hi
        
        mid = (lo + hi) // 2

        if particle_z == vertex_list[mid][1]:
            return mid
        elif particle_z > vertex_list[mid][1]:
            hi = mid
        else:
            lo = mid + 1
    return lo

################
# GRAPH OBJECT #
################

class Graph:
    """
    Class for creating undirected graphs. 
    Graphs are represented in the following way: Vertices are represented by a 
    dictionary indexed by numbers, beginning at 0. For each vertex, its value consists of the size of the vertex,
    its neighbors, and its coordinates. The edges are represented by a set of frozen sets, where each set has two vertices. Order of
    the vertices do not matter.
    """
    def __init__(self):
        """
        Creates an instance of the Graph Class. The graph is empty
        
        """
        self.vertices = {}
        self.edges = set()
        self.total_vertices = 0
        self.coordinate_containers = {}
        self.boundary_size = 0
        self.partition_length = 0
        self.smallest_particle = 0
        self.num_divisions = 0
        self.exact_volume = 0

    def add_vertex(self, vertex_size):
        """
        Input: a size of the vertex
        """
    
        self.vertices[self.total_vertices] = {'size': vertex_size, 'neighbors': set(), 'coordinates': None}
        self.total_vertices += 1

    def add_edge(self, edge):
        """
        Input: a set with two elements. The two 
        elements are both integers representing vertices to be connected by an edge.
        
        self.edges is updated to reflect the new edges. self.vertices is updated to 
        reflect new neighbors
        """
        if edge[0] not in self.vertices or edge[1] not in self.vertices:
            raise VertexNotPresentError('Vertices must be present in instance!')
        elif len(edge) != 2:
            raise ImproperEdgeError('Edges must contain two vertices!')
        self.edges.add(frozenset(edge))
        self.vertices[edge[0]]['neighbors'].add(edge[1])
        self.vertices[edge[1]]['neighbors'].add(edge[0])
    
    def get_coordinates(self, vertex_index):
        """
        Gets coordinates of a specified vertex
        """
        return self.vertices[vertex_index]['coordinates']

    def check_neighbors(self, vertex):
        """
        Input: an integer index of a vertex
        Returns: neighbors
        """
        return self.vertices[vertex]['neighbors']

    def assign_coordinate(self, vertex_index, coordinates):
        """
        Input: 
            vertex: an integer index of a vertex
            coordinates: a list of three coordinates representing [x,y,z] in R^3.
            assign_containers: True or False, depending on whether one wants to assign
            coordinate containers to vertex upon assign the coordinates

        Returns: Nothing, assigns coordinate to vertex in self.vertices
        """
        self.vertices[vertex_index]['coordinates'] = coordinates

        # Logic for adding particle to coordinate container
        partition_length = self.partition_length
        particle_size, particle_coordinates = self.vertices[vertex_index]['size'], self.vertices[vertex_index]['coordinates'] 

        containers = find_relevant_containers(self, particle_size, particle_coordinates)
        particle_z = self.get_coordinates(vertex_index)[2]
        for container in containers:
            vertex_list = self.coordinate_containers[container]
            vertex_list.insert(bisect_search(vertex_list, particle_size + particle_z), (vertex_index, particle_z + particle_size, particle_z - particle_size))


    def distance(self, vertex1, vertex2):
        """
        Input: The integer indices of two vertices
        returns: The distance between them if the positions of both vertices are well-defined, 
        Exception otherwise
        """
        position1 = self.vertices[vertex1]['coordinates']
        position2 = self.vertices[vertex2]['coordinates']
        if position1 == None or position2 == None:
            raise PositionNotDefinedError
        else:
            return math.dist(position1, position2)
        
        
if __name__ == "__main__" :
    sample_particles = [(.22, 1), (1.3, 1), (.002, 1)]
    sampleGraph = Graph(sample_particles)
    print('----------------------------------------------------')
    print(sampleGraph.vertices)
    print(sampleGraph.edges)
    print('----------------------------------------------------')
    sampleGraph.add_edge((0,1))
    sampleGraph.assign_coordinate(1, [1.05, 123.123, 123.12])
    sampleGraph.add_edge((1,2))
    print(sampleGraph.check_neighbors(1))
    print(sampleGraph.vertices)
    print(sampleGraph.edges)
    sampleGraph.assign_all_coordinates()



