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
        
    print("\n--- Joint Vertex Angle Detection ---")
    print(f"{'Joint (Vertex)':<15} | {'Connected Arms':<18} | {'Interior Angle (°)'}")
    print("-" * 55)
    
    for node_c, neighbors in sorted(graph.items()):
        neighbors = list(neighbors)
       
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
                    
                
                cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
                angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                
                print(f"Node N{node_c:<10} | N{n_a} -- N{node_c} -- N{n_b:<3}   | {angle_deg:6.1f}°")
                
    print("-" * 55)
def draw_vertex_angle(ax, nodes, center_idx, node_a_idx, node_b_idx, radius=0.35):
    c = nodes[center_idx]
    a = nodes[node_a_idx]
    b = nodes[node_b_idx]

    v1 = a - c
    v2 = b - c

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return

    ang1 = np.degrees(np.arctan2(v1[1], v1[0]))
    ang2 = np.degrees(np.arctan2(v2[1], v2[0]))

    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
    interior_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    if interior_angle < 0.5:
        return

    angle_delta = (ang2 - ang1 + 180) % 360 - 180
    theta1 = ang1
    theta2 = ang1 + angle_delta
    if theta2 < theta1:
        theta1, theta2 = theta2, theta1
    arc = patches.Arc(
        (c[0], c[1]),
        width=radius * 2,
        height=radius * 2,
        angle=0,
        theta1=theta1,
        theta2=theta2,
        color="purple",
        linewidth=1.2,
        zorder=5,
    )
    ax.add_patch(arc)

    mid_angle = np.radians((theta1 + theta2) / 2.0)
    text_radius = radius + 0.18
    text_x = c[0] + text_radius * np.cos(mid_angle)
    text_y = c[1] + text_radius * np.sin(mid_angle)

    ax.text(
        text_x,
        text_y,
        f"{interior_angle:.1f}°",
        color="purple",
        fontsize=6,
        weight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.85),
        zorder=6,
    )

def plot_stress_graph(
    nodes,
    elements,
    forces,
    title="Estimated member forces",
):
    columns = 5
    rows = max(1, math.ceil(len(elements) / columns))
    summary_height = 0.62 * rows
    fig = plt.figure(figsize=(12, 6.2 + summary_height), dpi=150)
    grid = fig.add_gridspec(
        2, 1, height_ratios=(summary_height, 6.2), hspace=0.08
    )
    summary_ax = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1])
    summary_ax.set_axis_off()
    summary_ax.set_title("Member details", fontsize=10, pad=2)

    for idx, (n1, n2) in enumerate(elements):
        p1, p2 = nodes[n1], nodes[n2]
        force = forces[idx]

        delta = p2 - p1
        member_length = np.linalg.norm(delta)

        if force > 0:
            color, state = "tab:blue", "T"
        elif force < 0:
            color, state = "tab:red", "C"
        else:
            color, state = "gray", "0"

        row, column = divmod(idx, columns)
        summary_ax.text(
            (column + 0.5) / columns,
            1.0 - (row + 0.5) / rows,
            f"N{n1} - N{n2}\n"
            f"{member_length:.2f} in | {abs(force):.1f} lbf {state}",
            transform=summary_ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=color,
                edgecolor=color,
                alpha=0.12,
            ),
        )

        if member_length == 0:
            continue

        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color=color, linewidth=2.5, zorder=2,
        )

    joint_graph = defaultdict(set)
    for n1, n2 in elements:
        joint_graph[n1].add(n2)
        joint_graph[n2].add(n1)

    for center_idx, neighbors in joint_graph.items():
        center = nodes[center_idx]
        ordered_neighbors = sorted(
            (
                (
                    np.degrees(
                        np.arctan2(
                            nodes[neighbor][1] - center[1],
                            nodes[neighbor][0] - center[0],
                        )
                    ),
                    neighbor,
                )
                for neighbor in neighbors
            ),
            key=lambda item: item[0],
        )
        if len(ordered_neighbors) == 2:
            draw_vertex_angle(
                ax,
                nodes,
                center_idx,
                ordered_neighbors[0][1],
                ordered_neighbors[1][1],
                radius=0.32,
            )
            continue

        for i, (angle_a, node_a_idx) in enumerate(ordered_neighbors):
            angle_b, node_b_idx = ordered_neighbors[(i + 1) % len(ordered_neighbors)]
            angle_gap = (angle_b - angle_a) % 360
            if angle_gap < 180:
                draw_vertex_angle(
                    ax,
                    nodes,
                    center_idx,
                    node_a_idx,
                    node_b_idx,
                    radius=0.32,
                )

    ax.scatter(nodes[:, 0], nodes[:, 1], color='black', s=45, zorder=3)
    for i, (x, y) in enumerate(nodes):
        ax.annotate(
            f"N{i}", (x, y), xytext=(4, 5), textcoords="offset points",
            fontsize=8, color="darkblue", weight="bold", zorder=5,
        )

    for node, marker, label in (
        (PINNED_SUPPORT_NODE, "^", "Pinned support"),
        (ROLLER_SUPPORT_NODE, "s", "Roller support"),
    ):
        if 0 <= node < len(nodes):
            ax.scatter(
                *nodes[node], marker=marker, s=110, facecolors="none",
                edgecolors="dimgray", linewidths=1.5, zorder=4, label=label,
            )

    ax.plot([], [], color="tab:blue", linewidth=2.5, label="Tension")
    ax.plot([], [], color="tab:red", linewidth=2.5, label="Compression")
    ax.plot([], [], color="gray", linewidth=2.5, label="Zero force")
    ax.set_title(title, weight="bold")
    ax.set_xlabel("Span Length (inches)")
    ax.set_ylabel("Height (inches)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.axis("equal")
    ax.margins(x=0.08, y=0.18)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.1, top=0.96)
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
    Print each member's end nodes, length, and angle in a table.
    """
    print("\n--- Member Geometry ---")
    print(f"{'Member (nodes)':^18} | {'Length (in)':>11} | {'Length (cm)':>11} | {'Angle (°)':>9}")
    print("-" * 62)

    for n1, n2 in elements:
        p1 = nodes[n1]
        p2 = nodes[n2]

        delta = p2 - p1
        length_in = np.linalg.norm(delta)
        length_cm = length_in * 2.54
        angle_rad = np.arctan2(delta[1], delta[0])
        angle_deg = np.degrees(angle_rad)

        print(
            f"{f'N{n1} - N{n2}':^18} | "
            f"{length_in:11.2f} | {length_cm:11.2f} | {angle_deg:9.1f}"
        )
    print("-" * 62)

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

def evaluate_fitness(nodes, elements):
    
    top_xs = [nodes[i, 0] for i in range(5, len(nodes))]
    if not all(top_xs[i] < top_xs[i+1] for i in range(len(top_xs) - 1)):
        return 999999.0

    if not check_connectivity(len(nodes), elements, SUPPORT_NODES):
        return 999999.0
        
    capacity_res = find_estimated_capacity(nodes, elements)
    if capacity_res is None:
        return 999999.0
        
    capacity_lbf, limiting_member, limiting_mode, unit_deflection, forces, lengths = capacity_res
    
    total_mass = sum(lengths)
    target_load_lbf = TEST_MARBLE_COUNT * marble_weight_lbf()
    
    capacity_penalty = 0.0
    if capacity_lbf < target_load_lbf:
        capacity_penalty = (target_load_lbf - capacity_lbf) * 200.0
        
    return total_mass + capacity_penalty


def create_random_individual():
   
    nodes, elements = create_truss()
    base_x = [2.0, 5.0, 9.0, 12.0]
    for idx, i in enumerate(range(5, len(nodes))):
        nodes[i, 1] = np.random.uniform(1.2, 4.0)
        nodes[i, 0] = base_x[idx] + np.random.uniform(-0.6, 0.6)
    return nodes, elements


def mutate(nodes):
   
    mutated_nodes = nodes.copy()
    for i in range(5, len(mutated_nodes)):
        if np.random.rand() < MUTATION_RATE:
            mutated_nodes[i, 1] += np.random.normal(0, 0.3)
            mutated_nodes[i, 0] += np.random.normal(0, 0.3)
            
            mutated_nodes[i, 1] = np.clip(mutated_nodes[i, 1], 1.0, 4.5)
            
            mutated_nodes[i, 0] = np.clip(mutated_nodes[i, 0], 0.5, 13.5)
            
    return mutated_nodes


def crossover(parent1_nodes, parent2_nodes):
   
    child_nodes = parent1_nodes.copy()
    for i in range(len(child_nodes)):
        if np.random.rand() > 0.5:
            child_nodes[i] = parent2_nodes[i]
    _, elements = create_truss()
    return child_nodes, elements


def run_genetic_algorithm():
    
    
    population = [create_random_individual() for _ in range(POPULATION_SIZE)]
    
    best_overall_nodes = None
    best_overall_elements = None
    best_overall_fitness = math.inf
    
    for generation in range(NUM_GENERATIONS):
        scored_population = []
        for nodes, elements in population:
            fitness = evaluate_fitness(nodes, elements)
            scored_population.append((fitness, nodes, elements))
            
        scored_population.sort(key=lambda x: x[0])
        gen_best_fitness, gen_best_nodes, gen_best_elements = scored_population[0]
        
        if gen_best_fitness < best_overall_fitness:
            best_overall_fitness = gen_best_fitness
            best_overall_nodes = gen_best_nodes.copy()
            best_overall_elements = gen_best_elements
            
        print(f"Generation {generation + 1:2d} | Best Fitness Score: {gen_best_fitness:.2f}")
        
        survivor_count = max(2, int(POPULATION_SIZE * 0.3))
        survivors = [item[1:] for item in scored_population[:survivor_count]]
        
        next_generation = list(survivors)
        
        while len(next_generation) < POPULATION_SIZE:
            p1 = survivors[np.random.choice(len(survivors))]
            p2 = survivors[np.random.choice(len(survivors))]
            
            child_nodes, child_elements = crossover(p1[0], p2[0])
            child_nodes = mutate(child_nodes)
            next_generation.append((child_nodes, child_elements))
            
        population = next_generation
    return best_overall_nodes, best_overall_elements




# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys
    marble_weight = marble_weight_lbf()

    
    opt_nodes, opt_elements = run_genetic_algorithm()

    if opt_nodes is None or opt_elements is None:
        print("Genetic algorithm did not produce a valid truss.")
        sys.exit(1)

    print_member_geometry(opt_nodes, opt_elements)
    detect_all_joint_angles(opt_nodes, opt_elements)

   
    if EXPLICIT_TEST_LOAD_LBF is not None:
        test_load_lbf = EXPLICIT_TEST_LOAD_LBF
        print(f"\nRunning simulation with explicit test load: {test_load_lbf} lbf")
    else:
        test_load_lbf = -(TEST_MARBLE_COUNT * marble_weight)
        print(f"\nRunning simulation with configured load: {TEST_MARBLE_COUNT} marbles ({abs(test_load_lbf):.3f} lbf)")

    
    forces, lengths, displacements = solve_truss(opt_nodes, opt_elements, test_load_lbf)

    if forces is None or lengths is None or displacements is None:
        print("Optimized structure failed or stiffness matrix is singular under this load.")
        sys.exit(1)

    print(f"Estimated marble weight: {marble_weight:.4f} lbf each")
    print(f"Total applied load: {abs(test_load_lbf):.3f} lbf ({abs(test_load_lbf) * NEWTONS_PER_LBF:.2f} N)")
    
    print("\nMember forces under test load:")
    for (n1, n2), force in zip(opt_elements, forces):
        state = "tension" if force > 0 else "compression"
        print(f"  N{n1} - N{n2}: {abs(force):8.3f} lbf {state}")

    center_deflection = displacements[2 * LOAD_NODE + 1]
    print(f"Predicted center deflection: {abs(center_deflection):.4f} inches")

   
    plot_stress_graph(
        opt_nodes,
        opt_elements,
        forces,
        title=f"Optimized Truss Bridge",
    )

    capacity = find_estimated_capacity(opt_nodes, opt_elements)
    if capacity is not None:
        capacity_lbf = capacity[0]
        capacity_marbles = capacity_lbf / marble_weight
        print(f"\n[Reference Limit] Maximum estimated bridge capacity: {capacity_lbf:.2f} lbf (~{capacity_marbles:.0f} marbles)")