import json
import networkx as nx


class LandmarkMap:
    def __init__(self, filename):
        with open(filename, "r") as file:
            self.data = json.load(file)

        self.graph = nx.Graph()
        self.build_graph()

    def build_graph(self):
        """
        Create a graph where every landmark is a node.
        Neighboring landmarks are connected by an edges.
        """
        for landmark in self.data["landmarks"]:
            self.graph.add_node(
                landmark["id"],
                id=landmark["id"],
                row=landmark["row"],
                column=landmark["column"],
                name=landmark["name"],
                type=landmark["type"],
                tags=landmark["tags"],
            )

        # Create graph edges.
        nodes = list(self.graph.nodes)

        for source in nodes:
            row1 = self.graph.nodes[source]["row"]
            column1 = self.graph.nodes[source]["column"]

            for target in nodes:
                if source == target:
                    continue

                row2 = self.graph.nodes[target]["row"]
                column2 = self.graph.nodes[target]["column"]

                dr = abs(row2 - row1)
                dc = abs(column2 - column1)

                if dr + dc == 1:
                    self.graph.add_edge(source, target, weight=1)

    def get_landmark(self, landmark_id):
        """
        Return all information about a landmark.
        Returns None if the landmark does not exist.
        """
        if not self.graph.has_node(landmark_id):
            return None

        return self.graph.nodes[landmark_id]

    def find_landmark(self, tag_id):
        """
        Find which landmark contains the detected AprilTag.

        Returns
        {
            "landmark": node_data,
            "position": "north"
        }
        """
        for landmark_id in self.graph.nodes:
            landmark = self.graph.nodes[landmark_id]

            for position, tag in landmark["tags"].items():
                if tag == tag_id:
                    return {
                        "landmark": landmark,
                        "id": landmark_id,
                        "position": position,
                    }

        return None

    def get_neighbors(self, landmark_id):
        """
        Return all neighboring landmarks.
        Returns
        list[int]
            List of neighboring landmark IDs.
        """
        if not self.graph.has_node(landmark_id):
            return []

        return list(self.graph.neighbors(landmark_id))

    def find_path(self, start_id, goal_id):
        """
        Compute the shortest path between two landmarks using A*.

        Returns
        list[int]
            List of landmark IDs from start to goal.
        """
        if not self.graph.has_node(start_id):
            return []

        if not self.graph.has_node(goal_id):
            return []

        try:
            return nx.astar_path(
                self.graph,
                start_id,
                goal_id,
                heuristic=self.heuristic,
                weight="weight",
            )

        except nx.NetworkXNoPath:
            return []

    def heuristic(self, current_id, goal_id):
        """
        Manhatten distance heuristic used by A*.
        """
        current = self.get_landmark(current_id)
        goal = self.get_landmark(goal_id)

        if current is None or goal is None:
            return float("inf")

        dr = abs(goal["row"] - current["row"])
        dc = abs(goal["column"] - current["column"])

        return dr + dc

    def get_next_landmark(self, path, path_index):
        """
        Return the immediate next landmark toward the goal.
        """
        if path_index + 1 >= len(path):
            return None

        return path[path_index + 1]

    def get_heading(self, current_id, target_id):
        """
        Return the desired heading from one landmark to another.

        Heading Convention
        ------------------
        North :   0°
        East  :  90°
        South : 180°
        West  : 270°
        """

        current = self.get_landmark(current_id)
        target = self.get_landmark(target_id)

        if current is None or target is None:
            return None

        dr = target["row"] - current["row"]
        dc = target["column"] - current["column"]

        # North
        if dr == 1 and dc == 0:
            return 0.0

        # East
        if dr == 0 and dc == 1:
            return 90.0

        # South
        if dr == -1 and dc == 0:
            return 180.0

        # West
        if dr == 0 and dc == -1:
            return -90.0

        raise ValueError(f"Landmarks {current_id} and {target_id} are not neighbors.")
