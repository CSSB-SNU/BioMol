import ast
import re


class ContactGraph:
    def __init__(self, graph_path: str):
        self.graphs: dict[
            tuple, dict[str, object]
        ] = {}  # key: header tuple, value: dict with vertices and edges
        self._parse_graph(graph_path)

    def _parse_graph(self, graph_path: str):
        """
        Parse a graph file that may contain multiple graph sections.
        Each section starts with a header line like:
            t # ('1', '1', '.')
        This header is used as a key, and the following lines define vertices and edges until the next header.

        Vertex line format:
            v <vertex_id> <vertex_value>

        Edge line format:
            e <vertex1> <vertex2>
        """  # noqa: E501
        current_header = None
        current_vertices: dict[int, int] = {}
        current_edges: list[tuple[int, int]] = []

        with open(graph_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # skip empty lines

                parts = line.split()
                line_type = parts[0]

                if line_type == "t":
                    # When encountering a new header, save the current graph (if any)
                    if current_header is not None:
                        self.graphs[current_header] = {
                            "vertices": current_vertices,
                            "edges": current_edges,
                        }

                    # Extract header information after the '#'
                    # Example line: t # ('1', '1', '.')
                    header_match = re.match(r"t\s+#\s+(.*)", line)
                    if header_match:
                        header_str = header_match.group(1)
                        try:
                            header_key = ast.literal_eval(header_str)
                        except Exception as e:
                            raise ValueError(
                                f"Could not parse header key: {header_str} \
                                    in line: {line}"
                            ) from e
                    else:
                        raise ValueError(f"Header line in unexpected format: {line}")

                    current_header = header_key
                    # Reset vertices and edges for the new graph section
                    current_vertices = {}
                    current_edges = []

                elif line_type == "v":
                    # Expecting: v <vertex_id> <vertex_value>
                    if len(parts) < 3:
                        raise ValueError(f"Invalid vertex line: {line}")
                    vertex_id = int(parts[1])
                    vertex_val = int(parts[2])
                    current_vertices[vertex_id] = vertex_val

                elif line_type == "e":
                    # Expecting: e <vertex1> <vertex2>
                    if len(parts) < 3:
                        raise ValueError(f"Invalid edge line: {line}")
                    vertex1 = int(parts[1])
                    vertex2 = int(parts[2])
                    current_edges.append((vertex1, vertex2))

                else:
                    raise ValueError(f"Unknown line type in line: {line}")

        # Save the last graph section after the loop ends
        if current_header is not None:
            self.graphs[current_header] = {
                "vertices": current_vertices,
                "edges": current_edges,
            }

    def filter_vertices(self, vertex_ids: list[int]):
        """
        Filter the vertices of the current graph section by a list of vertex IDs.
        This modifies the current graph section to only include the specified vertices.
        """
        if self.id is None:
            raise ValueError("No graph section selected. Use choose() to select one.")
        
        graph_data = self.graphs.get(self.id, {})
        vertices = graph_data.get("vertices", {})
        edges = graph_data.get("edges", [])

        # Filter vertices
        filtered_vertices = {k: v for k, v in vertices.items() if k in vertex_ids}
        
        # Filter edges based on the filtered vertices
        filtered_edges = [
            (v1, v2) for v1, v2 in edges if v1 in filtered_vertices and v2 in filtered_vertices
        ]

        # Update the current graph section
        self.graphs[self.id] = {
            "vertices": filtered_vertices,
            "edges": filtered_edges,
        }

    def choose(self, id: tuple[str, str, str]) -> dict[str, object]:
        """
        Choose a graph section by its header tuple
        """
        self.id = id

    def get_contact_node(self, id: tuple[str, str, str], node: int) -> list[int]:
        """
        Get contact nodes that are connected to the input node.

        This implementation iterates through all graph sections. For each section in which
        the node is defined, it examines the edges and collects any nodes that are connected
        to the given node.
        """  # noqa: E501
        if id is None:
            id = self.id
        contact_nodes = set()
        # Iterate over each graph section
        graph_data = self.graphs.get(id, {})
        vertices: dict[int, int] = graph_data.get("vertices", {})
        edges: list[tuple[int, int]] = graph_data.get("edges", [])
        # Only proceed if the node exists in this section
        if node in vertices:
            for edge in edges:
                if edge[0] == node:
                    contact_nodes.add(edge[1])
                elif edge[1] == node:
                    contact_nodes.add(edge[0])
        return list(contact_nodes)


if __name__ == "__main__":
    graph_path = "/data/psk6950/PDB_2024Mar18/protein_graph/tk/5tkk.graph"
    contact_graph = ContactGraph(graph_path)
    breakpoint()
