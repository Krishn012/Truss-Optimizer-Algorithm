import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import matplotlib.patches as patches
import sys
# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

# 1. Material Properties (Birch Wood)
WOOD_MODULUS_PSI = 1_300_000.0
ESTIMATED_TENSILE_STRENGTH_PSI = 8_000.0

# 2. Cross-Section Dimensions & Assembly
STICK_WIDTH_IN = 5.0 / 2.54
STICK_THICKNESS_IN = 1.0 / 25.4
STICKS_PER_MEMBER = 1

# 3. Test Load & Marble Parameters
MARBLE_MASS_GRAMS = 4.5
TEST_MARBLE_COUNT = 50.0   

EXPLICIT_TEST_LOAD_LBF = None 

# 4. Physical & Environmental Constants
GRAVITY_M_S2 = 9.80665
NEWTONS_PER_LBF = 4.448221615
SAFETY_FACTOR = 2.0
EULER_EFFECTIVE_LENGTH_FACTOR = 1.0

# 5. Solver & Structural Node Mapping
PINNED_SUPPORT_NODE = 0
ROLLER_SUPPORT_NODE = 4
SUPPORT_NODES = (PINNED_SUPPORT_NODE, ROLLER_SUPPORT_NODE)
LOAD_NODE = 2
MAX_GRAPH_POINTS = 100

# 6. Genetic Algorithm

POPULATION_SIZE = 30
NUM_GENERATIONS = 20
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.3

# ==============================================================================
# DERIVED CONSTANTS
# ==============================================================================

AREA = STICK_WIDTH_IN * STICK_THICKNESS_IN * STICKS_PER_MEMBER
I_MOMENT = (
    STICK_WIDTH_IN * STICK_THICKNESS_IN**3 / 12.0 * STICKS_PER_MEMBER
)
EA = WOOD_MODULUS_PSI * AREA

# ==============================================================================
# CORE PHYSICS & SOLVER FUNCTIONS
# ==============================================================================

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

def detect_all_joint_angles(nodes, elements):
    
    
    graph = defaultdict(set)
    for n1, n2 in elements:
        graph[n1].add(n2)
        graph[n2].add(n1)
        
    print("\n--- Programmatic Joint Vertex Angle Detection ---")
    print(f"{'Joint (Vertex)':<15} | {'Connected Arms':<18} | {'Interior Angle (°)'}")
    print("-" * 55)
    
    for node_c, neighbors in sorted(graph.items()):
        neighbors = list(neighbors)
        # If a node has at least 2 neighbors, we can compute angles between them
        if len(neighbors) < 2:
            continue
            
        c_pos = nodes[node_c]
        
       
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                n_a = neighbors[i]
                n_b = neighbors[j]
                
                v1 = nodes[n_a] - c_pos
                v2 = nodes[n_b] - c_pos
                
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                
                if norm_v1 == 0 or norm_v2 == 0:
                    continue
                    
                # Calculate interior angle using the dot product
                cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
                angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                
                print(f"Node N{node_c:<10} | N{n_a} -- N{node_c} -- N{n_b:<3}   | {angle_deg:6.1f}°")
                
    print("-" * 55)
def draw_vertex_angle(ax, nodes, center_idx, node_a_idx, node_b_idx, radius=0.6, label_offset=(0.0, 0.2)):
    """
    Draws a true geometric angle arc at center_idx between node_a and node_b (e.g., angle N0-N1-N5).
    """
    c = nodes[center_idx]
    a = nodes[node_a_idx]
    b = nodes[node_b_idx]
    
    v1 = a - c
    v2 = b - c
    
    ang1 = np.degrees(np.arctan2(v1[1], v1[0]))
    ang2 = np.degrees(np.arctan2(v2[1], v2[0]))
    
    theta1 = min(ang1, ang2)
    theta2 = max(ang1, ang2)
    if theta2 - theta1 > 180:
        theta1, theta2 = theta2, theta1 + 360
        
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    interior_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    arc = patches.Arc(
        (c[0], c[1]), 
        width=radius*2, height=radius*2, 
        angle=0, 
        theta1=theta1, 
        theta2=theta2, 
        color='purple', 
        linewidth=1.5, 
        linestyle='--',
        zorder=5
    )
    ax.add_patch(arc)
    
    mid_angle = np.radians((theta1 + theta2) / 2.0)
    text_x = c[0] + (radius + 0.25) * np.cos(mid_angle) + label_offset[0]
    text_y = c[1] + (radius + 0.25) * np.sin(mid_angle) + label_offset[1]
    
    ax.text(
        text_x, text_y, 
        f"{interior_angle:.1f}°", 
        color='purple', 
        fontsize=8, 
        weight='bold',
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8),
        zorder=6
    )

def plot_stress_graph(
    nodes,
    elements,
    forces,
    member_lengths,
    title="Estimated member forces",
    applied_load=None,
):
    plt.figure(figsize=(14, 7))
    ax = plt.gca()
    
    
    for idx, (n1, n2) in enumerate(elements):
        p1, p2 = nodes[n1], nodes[n2]
        force = forces[idx]
        length = member_lengths[idx]
        
        delta = p2 - p1
        angle_deg = np.degrees(np.arctan2(delta[1], delta[0]))
        
        color = 'red' if force < 0 else 'blue'
        linewidth = min(max(abs(force) / 6.0, 2), 8)
        
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth, zorder=2)
        
        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0
        normal = np.array([-delta[1], delta[0]]) / length
        label_x, label_y = np.array([mid_x, mid_y]) + 0.18 * normal
        state = "T" if force > 0 else "C"
        
        label_text = f"M{idx + 1}: {length:.2f}in ({angle_deg:.0f}°)\n{abs(force):.2f} lbf {state}"
        
        plt.text(
            label_x,
            label_y,
            label_text,
            fontsize=7,
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9),
            zorder=4,
        )

    
    graph = defaultdict(set)
    for n1, n2 in elements:
        graph[n1].add(n2)
        graph[n2].add(n1)
        
    for node_c, neighbors in graph.items():
        neighbors = list(neighbors)
        if len(neighbors) < 2:
            continue
            
        c_pos = nodes[node_c]
        
        
        neighbor_angles = []
        for n_nbr in neighbors:
            v = nodes[n_nbr] - c_pos
            ang = np.degrees(np.arctan2(v[1], v[0]))
            neighbor_angles.append((ang, n_nbr))
            
       
        neighbor_angles.sort(key=lambda x: x[0])
        
        
        num_n = len(neighbor_angles)
        for i in range(num_n):
            ang1, n_a = neighbor_angles[i]
            ang2, n_b = neighbor_angles[(i + 1) % num_n] 
            
            a = nodes[n_a]
            b = nodes[n_b]
            
            v1 = a - c_pos
            v2 = b - c_pos
            
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 == 0 or norm_v2 == 0:
                continue
                
            cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
            interior_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            
           
            if interior_angle < 85:
                theta1 = min(ang1, ang2)
                theta2 = max(ang1, ang2)
                if theta2 - theta1 > 180:
                    theta1, theta2 = theta2, theta1 + 360
                    
                radius = 0.4
                arc = patches.Arc(
                    (c_pos[0], c_pos[1]), 
                    width=radius*2, height=radius*2, 
                    angle=0, 
                    theta1=theta1, 
                    theta2=theta2, 
                    color='purple', 
                    linewidth=1.2, 
                    linestyle='--',
                    zorder=5
                )
                ax.add_patch(arc)
                
                mid_angle = np.radians((theta1 + theta2) / 2.0)
                text_x = c_pos[0] + (radius + 0.2) * np.cos(mid_angle)
                text_y = c_pos[1] + (radius + 0.2) * np.sin(mid_angle)
                
                ax.text(
                    text_x, text_y, 
                    f"{interior_angle:.0f}°", 
                    color='purple', 
                    fontsize=7, 
                    weight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.05', fc='white', alpha=0.7),
                    zorder=6
                )

    
    plt.scatter(nodes[:, 0], nodes[:, 1], color='black', s=80, zorder=3)
    
    for i, (x, y) in enumerate(nodes):
        plt.text(x, y + 0.2, f"N{i}", fontsize=9, ha='center', color='darkblue', weight='bold')

   

    plt.title(f"{title} (Blue = Tension, Red = Compression)", fontsize=13, weight='bold')
    plt.xlabel("Span Length (inches)", fontsize=11)
    plt.ylabel("Height (inches)", fontsize=11)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')
    plt.show()

def solve_truss(nodes, elements, load_force, load_node=LOAD_NODE):
    num_nodes = len(nodes)
    
    if load_node < 0 or load_node >= num_nodes:
        return None, None, None

    if not check_connectivity(num_nodes, elements, SUPPORT_NODES):
        return None, None, None

    num_dofs = 2 * num_nodes 
    K_global = np.zeros((num_dofs, num_dofs))

    member_lengths = []
    for n1, n2 in elements:
        p1, p2 = nodes[n1], nodes[n2]
        L = np.linalg.norm(p2 - p1)
        member_lengths.append(L)
        if L == 0:
            return None, None, None
        
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
    F[2 * load_node + 1] = load_force

    constrained_dofs = [
        2 * PINNED_SUPPORT_NODE,
        2 * PINNED_SUPPORT_NODE + 1,
        2 * ROLLER_SUPPORT_NODE + 1,
    ]
    free_dofs = [i for i in range(num_dofs) if i not in constrained_dofs]

    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]
    
    try:
        if np.linalg.matrix_rank(K_ff) < len(free_dofs):
            return None, None, None
        u_free = np.linalg.solve(K_ff, F_f)
    except np.linalg.LinAlgError:
        return None, None, None

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

    return member_forces, member_lengths, u

def physic_slv(nodes, elements, load_force, load_node=LOAD_NODE):
    forces, lengths, _ = solve_truss(nodes, elements, load_force, load_node)
    return forces, lengths


def find_estimated_capacity(nodes, elements):
    unit_load = -1.0
    forces, lengths, displacements = solve_truss(nodes, elements, unit_load)
    if forces is None or lengths is None or displacements is None:
        return None

    lowest_capacity_lbf = math.inf
    limiting_member = None
    limiting_mode = None

    for idx, (force_per_lbf, length) in enumerate(zip(forces, lengths)):
        if force_per_lbf > 0:
            member_capacity = (
                ESTIMATED_TENSILE_STRENGTH_PSI * AREA / SAFETY_FACTOR
            )
            capacity_lbf = member_capacity / force_per_lbf
            mode = "estimated tension"
        elif force_per_lbf < 0:
            effective_length = EULER_EFFECTIVE_LENGTH_FACTOR * length
            member_capacity = (
                math.pi**2 * WOOD_MODULUS_PSI * I_MOMENT
                / effective_length**2 / SAFETY_FACTOR
            )
            capacity_lbf = member_capacity / abs(force_per_lbf)
            mode = "Euler buckling"
        else:
            continue

        if capacity_lbf < lowest_capacity_lbf:
            lowest_capacity_lbf = capacity_lbf
            limiting_member = idx
            limiting_mode = mode

    center_deflection_per_lbf = displacements[2 * LOAD_NODE + 1]
    return (
        lowest_capacity_lbf,
        limiting_member,
        limiting_mode,
        center_deflection_per_lbf,
        forces,
        lengths,
    )


def marble_weight_lbf():
    mass_kg = MARBLE_MASS_GRAMS / 1000.0
    return mass_kg * GRAVITY_M_S2 / NEWTONS_PER_LBF

def print_member_geometry(nodes, elements):
    """
    Programmatically calculates and prints the length, angle, 
    and connection nodes for every member in the truss.
    """
    print("\n--- Programmatic Member Geometry Breakdown ---")
    print(f"{'Member':<8} | {'Nodes':<10} | {'Length (in)':<12} | {'Angle (°)'}")
    print("-" * 48)
    
    for idx, (n1, n2) in enumerate(elements):
        p1 = nodes[n1]
        p2 = nodes[n2]
        
        delta = p2 - p1
        length = np.linalg.norm(delta)
        angle_rad = np.arctan2(delta[1], delta[0])
        angle_deg = np.degrees(angle_rad)
        
        print(f"M{idx+1:<7} | N{n1} -> N{n2:<5} | {length:10.2f} in  | {angle_deg:6.1f}°")
    print("-" * 48)
    
def plot_marble_response(capacity, marble_weight):
    capacity_lbf, limiting_member, limiting_mode = capacity[:3]
    unit_deflection = capacity[3]
    unit_forces = capacity[4]
    lengths = capacity[5]
    capacity_marbles = capacity_lbf / marble_weight
    max_plot_marbles = max(capacity_marbles * 1.15, 1.0)
    marble_counts = np.linspace(0.0, max_plot_marbles, MAX_GRAPH_POINTS)

    center_deflection_mm = (
        np.abs(unit_deflection) * marble_counts * marble_weight * 25.4
    )
    tension_capacity_lbf = (
        ESTIMATED_TENSILE_STRENGTH_PSI * AREA / SAFETY_FACTOR
    )
    tension_utilization_per_lbf = max(
        (force / tension_capacity_lbf for force in unit_forces if force > 0),
        default=0.0,
    )
    compression_utilization_per_lbf = 0.0
    for force, length in zip(unit_forces, lengths):
        if force < 0:
            compression_capacity_lbf = (
                math.pi**2 * WOOD_MODULUS_PSI * I_MOMENT
                / (EULER_EFFECTIVE_LENGTH_FACTOR * length) ** 2
                / SAFETY_FACTOR
            )
            compression_utilization_per_lbf = max(
                compression_utilization_per_lbf,
                abs(force) / compression_capacity_lbf,
            )
    tension_utilization = (
        tension_utilization_per_lbf * marble_counts * marble_weight
    )
    compression_utilization = (
        compression_utilization_per_lbf * marble_counts * marble_weight
    )

    fig, (deflection_ax, force_ax) = plt.subplots(2, 1, figsize=(10, 9))
    deflection_ax.plot(marble_counts, center_deflection_mm, color="darkgreen")
    deflection_ax.axvline(capacity_marbles, color="red", linestyle="--",
                          label=f"Estimated limit: {capacity_marbles:.0f} marbles")
    deflection_ax.set_title("Estimated bridge response to centered marble load")
    deflection_ax.set_ylabel("Center deflection (mm)")
    deflection_ax.set_xlabel("Marbles")
    deflection_ax.grid(True, linestyle=":", alpha=0.6)
    deflection_ax.legend()

    force_ax.plot(marble_counts, tension_utilization, color="blue",
                  label="Maximum tension utilization")
    force_ax.plot(marble_counts, compression_utilization, color="red",
                  label="Maximum buckling utilization")
    force_ax.axhline(1.0, color="black", linestyle=":",
                     label="Estimated limit (utilization = 1)")
    force_ax.axvline(capacity_marbles, color="black", linestyle="--")
    force_ax.set_ylabel("Demand / estimated capacity")
    force_ax.set_xlabel("Marbles")
    force_ax.grid(True, linestyle=":", alpha=0.6)
    force_ax.legend()

    fig.suptitle(
        f"First modeled limit: {limiting_mode}, member {limiting_member + 1}"
        if limiting_member is not None
        else "Bridge response estimate"
    )
    fig.tight_layout()
    plt.show()


# =============================================================================
# GENETIC ALGORITHM FUNCTIONS
# =============================================================================

def evaluate_fitness(nodes,elements):
    if not check_connectivity(len(nodes), elements, SUPPORT_NODES):
        return 999999.0


if __name__ == "__main__":
    
    nodes, elements = create_truss()
    marble_weight = marble_weight_lbf()
    print_member_geometry(nodes, elements)
    detect_all_joint_angles(nodes, elements)
    # Determine test load from top configuration variables
    if EXPLICIT_TEST_LOAD_LBF is not None:
        test_load_lbf = EXPLICIT_TEST_LOAD_LBF
        print(f"Running simulation with explicit test load: {test_load_lbf} lbf")
    else:
        test_load_lbf = -(TEST_MARBLE_COUNT * marble_weight)
        print(f"Running simulation with configured load: {TEST_MARBLE_COUNT} marbles ({abs(test_load_lbf):.3f} lbf)")

    # Run solver for the specific configured test load
    forces, lengths, displacements = solve_truss(nodes, elements, test_load_lbf)

    # Bulletproof check: if the solver failed, stop here cleanly
    if forces is None or displacements is None:
        print("Structure failed or stiffness matrix is singular under this load.")
        sys.exit()

    print(f"Estimated marble weight: {marble_weight:.4f} lbf each")
    print(f"Total applied load: {abs(test_load_lbf):.3f} lbf ({abs(test_load_lbf) * NEWTONS_PER_LBF:.2f} N)")
    
    print("Member forces under test load:")
    for idx, force in enumerate(forces, start=1):
        state = "tension" if force > 0 else "compression"
        print(f"  Member {idx:2d}: {abs(force):8.3f} lbf {state}")

    center_deflection = displacements[2 * LOAD_NODE + 1]
    print(f"Predicted center deflection: {abs(center_deflection):.4f} inches")

    # Plot stress distribution for this exact test load configuration
    plot_stress_graph(
        nodes,
        elements,
        forces,
        lengths,
        title=f"Bridge member forces under {abs(test_load_lbf):.1f} lbf load",
        applied_load=test_load_lbf,
    )

    capacity = find_estimated_capacity(nodes, elements)
    if capacity is not None:
        capacity_lbf = capacity[0]
        capacity_marbles = capacity_lbf / marble_weight
        print(f"\n[Reference Limit] Maximum estimated bridge capacity: {capacity_lbf:.2f} lbf (~{capacity_marbles:.0f} marbles)")