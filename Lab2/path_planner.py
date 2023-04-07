from grid import Node, NodeGrid
from math import inf
import heapq


class PathPlanner(object):
    """
    Represents a path planner, which may use Dijkstra, Greedy Search or A* to plan a path.
    """
    def __init__(self, cost_map):
        """
        Creates a new path planner for a given cost map.

        :param cost_map: cost used in this path planner.
        :type cost_map: CostMap.
        """
        self.cost_map = cost_map
        self.node_grid = NodeGrid(cost_map)

    @staticmethod
    def construct_path(goal_node):
        """
        Extracts the path after a planning was executed.

        :param goal_node: node of the grid where the goal was found.
        :type goal_node: Node.
        :return: the path as a sequence of (x, y) positions: [(x1,y1),(x2,y2),(x3,y3),...,(xn,yn)].
        :rtype: list of tuples.
        """
        node = goal_node
        # Since we are going from the goal node to the start node following the parents, we
        # are transversing the path in reverse
        reversed_path = []
        while node is not None:
            reversed_path.append(node.get_position())
            node = node.parent
        return reversed_path[::-1]  # This syntax creates the reverse list

    def dijkstra(self, start_position, goal_position):
        """
        Plans a path using the Dijkstra algorithm.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """

        pq = []
        nodes = {}
        nodes[start_position] = Node(*start_position)
        nodes[start_position].f = 0
        heapq.heappush(pq, (0, nodes[start_position]))

        while len(pq) != 0:

            f, node = heapq.heappop(pq)
            node.closed = True
            for successor in self.node_grid.get_successors(*node.get_position()):
                successor = nodes.setdefault(successor, Node(*successor))
                if successor.f > f + self.cost_map.get_edge_cost(node.get_position(), successor.get_position()):
                    successor.f = f + self.cost_map.get_edge_cost(node.get_position(), successor.get_position())
                    successor.parent = node
                    heapq.heappush(pq, (successor.f, successor))

        self.node_grid.reset()
        backtrack = self.construct_path(nodes[goal_position])
        final_cost = 0
        for node_i in range(len(backtrack)-1):
            final_cost += self.cost_map.get_edge_cost(nodes[backtrack[node_i]].get_position(), nodes[backtrack[node_i+1]].get_position())

        return backtrack, final_cost

    def greedy(self, start_position, goal_position):
        """
        Plans a path using greedy search.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """

        pq = []
        nodes = {}
        nodes[start_position] = Node(*start_position)
        nodes[start_position].f = nodes[start_position].distance_to(*goal_position)
        heapq.heappush(pq, (nodes[start_position].f, nodes[start_position]))

        found = False

        while len(pq) != 0 or not found:

            _, node = heapq.heappop(pq)
            node.closed = True
            for successor in self.node_grid.get_successors(*node.get_position()):

                successor = nodes.setdefault(successor, Node(*successor))
                if successor.f != inf:
                    continue
                successor.parent = node
                successor.f = successor.distance_to(*goal_position)
                if successor.get_position() == goal_position:
                    found = True
                heapq.heappush(pq, (successor.f, successor))

        self.node_grid.reset()

        backtrack = self.construct_path(nodes[goal_position])
        final_cost = 0
        for node_i in range(len(backtrack)-1):
            final_cost += self.cost_map.get_edge_cost(nodes[backtrack[node_i]].get_position(), nodes[backtrack[node_i+1]].get_position())

        return backtrack, final_cost

    def a_star(self, start_position, goal_position):
        """
        Plans a path using A*.

        :param start_position: position where the planning stars as a tuple (x, y).
        :type start_position: tuple.
        :param goal_position: goal position of the planning as a tuple (x, y).
        :type goal_position: tuple.
        :return: the path as a sequence of positions and the path cost.
        :rtype: list of tuples and float.
        """
        
        pq = []
        nodes = {}
        nodes[start_position] = Node(*start_position)
        nodes[start_position].f = nodes[start_position].distance_to(*goal_position)
        nodes[start_position].g = 0
        heapq.heappush(pq, (nodes[start_position].f, nodes[start_position]))

        found = False

        while len(pq) != 0 or not found:

            f, node = heapq.heappop(pq)
            node.closed = True

            if node.get_position() == goal_position:
                break
            
            for successor in self.node_grid.get_successors(*node.get_position()):
                successor = nodes.setdefault(successor, Node(*successor))
                if successor.f > node.g + self.cost_map.get_edge_cost(node.get_position(), successor.get_position()) + successor.distance_to(*goal_position):
                    successor.g = node.g + self.cost_map.get_edge_cost(node.get_position(), successor.get_position())
                    successor.f = successor.g + self.cost_map.get_edge_cost(node.get_position(), successor.get_position())
                    successor.parent = node
                    heapq.heappush(pq, (successor.f, successor))

        self.node_grid.reset()

        backtrack = self.construct_path(nodes[goal_position])

        final_cost = 0

        for node_i in range(len(backtrack)-1):
            final_cost += self.cost_map.get_edge_cost(nodes[backtrack[node_i]].get_position(), nodes[backtrack[node_i+1]].get_position())

        return backtrack, final_cost
