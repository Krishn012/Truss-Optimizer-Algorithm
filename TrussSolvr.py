import numpy as np
import matplotlib.pyplot as plt

def create_truss():
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
    
    # Plot one member
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
    num_nodes = len(nodes)
    num_dofs = 2*num_nodes 
    K_global = np.zeros((num_dofs, num_dofs)) # Blank global stiffness matrix
    for n1, n2 in elements:
        p1, p2 = nodes[n1], nodes[n2]
        L = np.linalg.norm(p2 - p1)
        if L == 0: continue
        c = (p2[0] - p1[0]) / L
        s = (p2[1] - p1[1]) / L
        
        k_local = (EA / L) * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])
        
        dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        for i in range(4):
            for j in range(4):
                K_global[dofs[i], dofs[j]] += k_local[i, j]

    # Apply Loads & Boundary Conditions (Pin left, roller right)
    F = np.zeros(num_dofs)
    # Find center node dynamically or assume index
    center_node = len(nodes) // 2
    F[2 * center_node + 1] = load_force

    constrained_dofs = [0, 1, 2*(num_nodes-1) + 1] # Left pinned (x,y), Right roller (y)
    free_dofs = [i for i in range(num_dofs) if i not in constrained_dofs]

    # Solve F = K * u
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]
    
    try:
        u_free = np.linalg.solve(K_ff, F_f)
    except np.linalg.LinAlgError:
        return None, None # Structure is unstable / collapses mathematically

    u = np.zeros(num_dofs)
    u[free_dofs] = u_free

    # Calculate Internal Forces
    member_forces = []
    for n1, n2 in elements:
        p1, p2 = nodes[n1], nodes[n2]
        L = np.linalg.norm(p2 - p1)
        c = (p2[0] - p1[0]) / L
        s = (p2[1] - p1[1]) / L
        u_element = np.array([u[2*n1], u[2*n1+1], u[2*n2], u[2*n2+1]])
        force = (EA / L) * np.dot(np.array([-c, -s, c, s]), u_element)
        member_forces.append(force)

    return member_forces, u

def evaluate_bridge(nodes, elements, load_force, max_allowable_stress=500.0):
    """
    Step 2: The Evaluator / Fitness Function
    Checks if the bridge breaks under load and calculates total material mass.
    """
    # 1. Calculate total length (proportional to mass)
    total_mass = sum(np.linalg.norm(nodes[n2] - nodes[n1]) for n1, n2 in elements)
    
    # 2. Run physics solver
    forces, displacements = physic_slv(nodes, elements, load_force)
    
    if forces is None:
        return 999999.0 # Penalty score for unstable geometry
        
    # 3. Check for structural failure (Stress / Buckling limit)
    failure_penalty = 0.0
    for force in forces:
        if abs(force) > max_allowable_stress:
            failure_penalty += 1000.0 # Heavy penalty if a stick breaks
            
    # 4. Final Fitness Score (Lower is better)
    fitness_score = total_mass + failure_penalty
    return fitness_score

if __name__ == "__main__":
    nodes, elements = create_truss()
    print(f"Truss created successfully with {len(nodes)} nodes and {len(elements)} members.")
    
    score = evaluate_bridge(nodes, elements, load_force=-50.0)
    print(f"Fitness Score: {score:.2f}")