import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


## CONSTANTS

E = 1_300_000.0
width = 0.375
thickness = 2.0 / 25.4
Area = width * thickness
I_moment = (width * (thickness ** 3)) / 12.0
EA = E * Area
TENSILE_STRENGTH = 8000.0


def check_connectivity(num_nodes, elements, constrained_nodes):
    graph = defaultdict(list)
    for n1, n2 in elements:
        graph[n1].append(n2)
        graph[n2].append(n1)
    
    visited = set()
    queue = list(constrained_nodes)
    for n in queue:
        visited.add(n)
        
    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                
    return len(visited) == num_nodes

def create_truss():
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

    elements = [
        # Bottom chord
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Top chord
        (5, 6), (6, 7), (7, 8),
        # Web diagonals and verticals
        (0, 5), (1, 5), (1, 6), (2, 6), 
        (2, 7), (3, 7), (3, 8), (4, 8)
    ]

    return nodes, elements

def plot_stress_graph(nodes, elements, forces, member_lengths):
    plt.figure(figsize=(12, 5))
    
    for idx, (n1, n2) in enumerate(elements):
        p1, p2 = nodes[n1], nodes[n2]
        force = forces[idx]
        L = member_lengths[idx]
        
        color = 'red' if force < 0 else 'blue'
        linewidth = min(max(abs(force) / 6.0, 2), 8)
        
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth, zorder=2)
        
        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0
        plt.text(mid_x, mid_y, f"{force:.1f} lbs", fontsize=8, ha='center', va='center', 
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85), zorder=4)

    plt.scatter(nodes[:, 0], nodes[:, 1], color='black', s=80, zorder=3)
    
    for i, (x, y) in enumerate(nodes):
        plt.text(x, y + 0.2, f"N{i}", fontsize=9, ha='center', color='darkblue', weight='bold')

    plt.title("Truss Force Map (Blue = Tension, Red = Compression)", fontsize=13, weight='bold')
    plt.xlabel("Span Length (inches)", fontsize=11)
    plt.ylabel("Height (inches)", fontsize=11)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')
    plt.show()

def physic_slv(nodes, elements, load_force):
    num_nodes = len(nodes)
    
    if not check_connectivity(num_nodes, elements, [0, num_nodes - 1]):
        return None, None

    num_dofs = 2 * num_nodes 
    K_global = np.zeros((num_dofs, num_dofs))

    member_lengths = []
    for n1, n2 in elements:
        p1, p2 = nodes[n1], nodes[n2]
        L = np.linalg.norm(p2 - p1)
        member_lengths.append(L)
        if L == 0: 
            return None, None
        
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

    F = np.zeros(num_dofs)
    center_node = len(nodes) // 2
    F[2 * center_node + 1] = load_force

    constrained_dofs = [0, 1, 2*(num_nodes-1) + 1] 
    free_dofs = [i for i in range(num_dofs) if i not in constrained_dofs]

    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]
    
    try:
        u_free = np.linalg.solve(K_ff, F_f)
    except np.linalg.LinAlgError:
        return None, None

    u = np.zeros(num_dofs)
    u[free_dofs] = u_free

    member_forces = []
    for idx, (n1, n2) in enumerate(elements):
        p1, p2 = nodes[n1], nodes[n2]
        L = member_lengths[idx]
        c = (p2[0] - p1[0]) / L
        s = (p2[1] - p1[1]) / L
        u_element = np.array([u[2*n1], u[2*n1+1], u[2*n2], u[2*n2+1]])
        force = (EA / L) * np.dot(np.array([-c, -s, c, s]), u_element)
        member_forces.append(force)

    return member_forces, member_lengths

def evaluate_bridge(nodes, elements, load_force=-50.0):
    forces, lengths = physic_slv(nodes, elements, load_force)
    
    if forces is None or lengths is None:
        return 999999.0
        
    total_mass = sum(lengths)
    failure_penalty = 0.0

    for force, L in zip(forces, lengths):
        if force > 0:
            stress = force / Area
            if stress > TENSILE_STRENGTH:
                failure_penalty += 5000.0
        else:
            if L > 0:
                p_cr = ((np.pi ** 2) * E * I_moment) / (L ** 2)
                if abs(force) > p_cr:
                    failure_penalty += 5000.0 + (abs(force) - p_cr) * 10.0

    return total_mass + failure_penalty


if __name__ == "__main__":
    nodes, elements = create_truss()
    
    # Test with a lighter load (-10 lbs) so you can clearly see forces mapped visually
    load = -10.0
    forces, lengths = physic_slv(nodes, elements, load_force=load)
    fitness = evaluate_bridge(nodes, elements, load_force=load)
    
    print(f"Fitness Score: {fitness:.2f}")
    
    if forces is not None:
        plot_stress_graph(nodes, elements, forces, lengths)
    else:
        print("Structure failed")