import numpy as np
import matplotlib.pyplot as plt

def create_sample_truss():
    """
    Creates a basic truss layout (similar to a Warren truss framework) 
    using coordinates scaled for standard popsicle sticks (~4.5 inches long).
    """
    # Define Nodes (x, y coordinates in inches)
    # Bottom chord nodes and top chord nodes
    nodes = np.array([
        [0.0, 0.0],   # Node 0: Bottom-left support
        [3.5, 0.0],   # Node 1: Bottom chord 
        [7.0, 0.0],   # Node 2: Bottom center (Loading point)
        [10.5, 0.0],  # Node 3: Bottom chord
        [14.0, 0.0],  # Node 4: Bottom-right support
        [1.75, 2.5],  # Node 5: Top chord left
        [5.25, 2.5],  # Node 6: Top chord mid-left
        [8.75, 2.5],  # Node 7: Top chord mid-right
        [12.25, 2.5]  # Node 8: Top chord right
    ])

    # Define Elements (pairs of node indices representing popsicle sticks)
    elements = [
        # Bottom chord
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Top chord
        (5, 6), (6, 7), (7, 8),
        # Web diagonals and verticals (forming triangles)
        (0, 5), (1, 5), (1, 6), (2, 6), 
        (2, 7), (3, 7), (3, 8), (4, 8)
    ]

    return nodes, elements

def plot_truss(nodes, elements, title="Bridge Blueprint"):
    """
    Visualizes the truss structure using Matplotlib.
    """
    plt.figure(figsize=(10, 4))
    
    # Plot each element (popsicle stick)
    for idx, (n1, n2) in enumerate(elements):
        p1 = nodes[n1]
        p2 = nodes[n2]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='saddlebrown', linewidth=4, zorder=2)
        
    # Plot nodes (joints)
    plt.scatter(nodes[:, 0], nodes[:, 1], color='black', s=80, zorder=3)
    
    # Label the nodes for easy debugging
    for i, (x, y) in enumerate(nodes):
        plt.text(x, y + 0.15, f"N{i}", fontsize=9, ha='center', color='blue', weight='bold')

    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel("Span Length (inches)", fontsize=11)
    plt.ylabel("Height (inches)", fontsize=11)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')
    plt.show()

def physic_slv(nodes, elements, load_force, EA=100):
    node_num = len(nodes)
    degrees_of_freedom = 2*node_num  # Each node has 2 DOF (x and y)
    K_global = np.zeros((degrees_of_freedom, degrees_of_freedom)) # Blank global stiffness matrix






if __name__ == "__main__":
    nodes, elements = create_sample_truss()
    print(f"Truss created successfully with {len(nodes)} nodes and {len(elements)} members.")
    plot_truss(nodes, elements, title="Initial Bridge Template (Warren-Style)")