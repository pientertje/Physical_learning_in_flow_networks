
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import time 
from matplotlib import animation
import matplotlib.collections as mc
from scipy.sparse.csgraph import connected_components
plt.close('all')
pi = np.pi
start = time.time() #starting time 
np.random.seed(2)  # Setting a seed to reproduce results
#%% inital conditions network
Nx, Ny = 20, 15 #size network
Nb = 6# edge nodes left and right
num_nodes = Nx * Ny  # Total internal nodes
#%% initial conditions simulation
pleft = 10  # bound pressure
pright = 0

q_left = 10
q_right = -10
mu = 1 #viscosity 

n= 2# constant for erosion power law
m =1
beta = 10/10 #clogging term

# conditions time and steps
t0 = 0
tmax = 100
dt = 0.01
t = t0

max_steps = 500
steps = 1

# the boundrys of the intrernal nodes 
left = 1
right = Nx-1

top = Ny -1/2
botom = 1/2
#%% setting up the network 

#internal nodes
x = np.random.uniform(left, right, num_nodes)
y = np.random.uniform(botom , top , num_nodes)

#generating uniform boundry nodes
y_left = np.linspace(botom, top, Nb)
y_right = y_left.copy()

x_left = np.zeros(Nb)
x_right = np.ones(Nb)*Nx

x_top = np.linspace(left, right, Nx)
x_bottom = x_top.copy()

y_top = np.ones(Nx)* Ny
y_bottom = np.zeros(Nx)

x = np.concatenate([x, x_top, x_bottom, x_left, x_right])
y = np.concatenate([y, y_top, y_bottom, y_left, y_right])

# generating nodes
nodes = np.column_stack((x, y))
num_nodes = len(nodes)

tri = scipy.spatial.Delaunay(nodes)  # Delaunay triangulation

# Extract edges from the triangulation
simplices = tri.simplices  # nodes of triangels 
edges = np.vstack([
    simplices[:, [0, 1]],
    simplices[:, [1, 2]],
    simplices[:, [2, 0]]
])  # each induvidial edge 

# Remove duplicate edges and sort
edges = np.unique(np.sort(edges, axis=1), axis=0)
num_edges = len(edges)

#finding internal and boundary nodes
boundary_nodes= np.where((x == x.min()) | (x == x.max()))[0] #boundry nodes
internal_nodes = np.setdiff1d(np.arange(num_nodes), boundary_nodes) # index internal nodes

#generating the top and botom dimond like part of the grid
top_boundary_nodes = np.where(y == y.max())[0]
bottom_boundary_nodes = np.where(y == y.min())[0]

# Find edges that connect top and bottom boundary nodes and remove them 
edges_to_remove = []
for n1 in top_boundary_nodes:
    for n2 in top_boundary_nodes:
        edges_to_remove.append([n1, n2])
        edges_to_remove.append([n2, n1])

for n1 in bottom_boundary_nodes:
    for n2 in bottom_boundary_nodes:
        edges_to_remove.append([n1, n2])
        edges_to_remove.append([n2, n1])
       
for n1 in boundary_nodes:
    for n2 in boundary_nodes:
        edges_to_remove.append([n1, n2])
        edges_to_remove.append([n2, n1])

for edge in edges_to_remove:
    edge_index = np.where((edges == edge).all(axis=1))[0]
    if edge_index.size > 0:
        edges = np.delete(edges, edge_index, axis=0)

# After this mapping, remove duplicate edges (if any)
edges = np.unique(np.sort(edges, axis=1), axis=0)
edges_periodic = edges.copy()

num_nodes = len(nodes)
num_edges = len(edges)


#setting up edge lenghts, edge widhts , edge lines
edge_lengths = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
edge_widths = np.random.uniform(5, 14, num_edges) # generating widht of the edges
edge_lines = np.array([(nodes[i], nodes[j]) for i, j in edges])


#finding the botom an top internal nodes 
top_in = np.where((nodes[:, 1] == Ny) & (nodes[:, 0] != 0) & (nodes[:, 0] != Nx))[0]
bot_in = np.where((nodes[:, 1] == 0 ) & (nodes[:, 0] != 0) & (nodes[:, 0] != Nx))[0]


for i, edge in enumerate(edges):
    n1 , n2 = edge
    if np.isin(n1, bot_in):
        top_index = np.where((nodes[:, 0] == nodes[:, 0][n1]) & (nodes[:, 1] == Ny))[0][0]
        edges[i][0]= top_index
        
    if np.isin(n2, bot_in):
        top_index = np.where((nodes[:, 0] == nodes[:, 0][n2]) & (nodes[:, 1] == Ny))[0][0]
        edges[i][1]= top_index

#%%  Extract submatrices

def D_matrix( edges,num_nodes, num_edges, boundary_nodes, internal_nodes):
    """function that generates the D , Db, Dn matrices"""
    D = np.zeros((num_edges, num_nodes))
    for i, (n1, n2) in enumerate(edges):
        D[i, n1] = 1
        D[i, n2] = -1

    Db = D[:, boundary_nodes]
    Dn = D[:, internal_nodes]
    return D, Db, Dn
    
D, Db, Dn = D_matrix( edges,num_nodes, num_edges, boundary_nodes, internal_nodes)

def removing_nodes():
    """Remove internal nodes that have only one edge and update all related arrays."""
    global x, y, D, Dn, nodes, edges, edge_widths, edge_lengths, edge_lines, boundary_nodes, C, q
    
    # Calculate the number of nodes
    num_nodes = len(nodes)
    changed = True 
    
    while changed:
        changed = False # remvoving lonley nodes itiratiavly cause removing can 
        #stil leave lonley nodes
        # Count the number of edges per node
        edge_count = np.zeros(num_nodes)
        for edge in edges:
            edge_count[edge[0]] += 1
            edge_count[edge[1]] += 1
            
        # Identify lonely internal nodes (excluding boundary nodes)
        internal_mask = ~np.isin(np.arange(num_nodes), boundary_nodes)
        lonely_nodes = np.where((edge_count <= 1) & internal_mask)[0]

        if len(lonely_nodes) > 0:
            changed = True
            # Create a mapping from old node indices to new node indices
            valid_nodes = np.setdiff1d(np.arange(num_nodes), lonely_nodes)
            node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_nodes)}
            
            # Remove lonely nodes from the node list
            nodes = nodes[valid_nodes]
            x = x[valid_nodes]
            y = y[valid_nodes]
            
            # Update boundary and internal nodes
            boundary_nodes = np.where((x == x.min()) | (x == x.max()))[0]
            internal_nodes = np.setdiff1d(np.arange(len(nodes)), boundary_nodes)
            
            # Remove edges connected to lonely nodes and update edge arrays
            valid_edges = ~np.isin(edges, lonely_nodes).any(axis=1)
            edges = edges[valid_edges]
            edge_widths = edge_widths[valid_edges]
            edge_lengths = edge_lengths[valid_edges]
            edge_lines = edge_lines[valid_edges]
           
         
            # Reindex the edges array using the new node indices
            edges = np.array([[node_mapping[n1], node_mapping[n2]] for n1, n2 in edges])
            
            num_nodes = len(nodes)
    
    # Rebuild the D matrix
    num_edges = len(edges)
    D, Db, Dn = D_matrix( edges,num_nodes, num_edges, boundary_nodes, internal_nodes)

        
    return x, y, D, Dn, Db, nodes, edges, edge_widths, edge_lengths, edge_lines, internal_nodes, boundary_nodes, num_nodes
# Call the function
x, y, D, Dn, Db, nodes, edges, edge_widths, edge_lengths, edge_lines, internal_nodes, boundary_nodes, num_nodes = removing_nodes()


def conductance(edge_widths):
    """function that calucalte the diagional conductence matrix"""
    C =  np.diag((pi*edge_widths**4)/(8* mu* edge_lengths))
    return C

def A_matrices( C):   
    Abb = Db.T @ C @ Db
    Abn = Db.T @ C @ Dn
    Anb = Dn.T @ C @ Db
    Ann = Dn.T @ C @ Dn
    return  Abb, Abn, Anb, Ann
#%%

def clean_graph():
    """Removes edges with width <= 1e-12, deletes lonely nodes, and removes isolated islands."""
    global q, C, x, y, D, Dn, Db, nodes, edges, edge_widths, edge_lengths, edge_lines, internal_nodes, boundary_nodes, num_nodes, steps

   #storing for plotting after detecition of removing islands
    q0 = q.copy()
    nodes0 = nodes.copy()
    edges0 = edges.copy()
    
    # vinding the internal smal edges 
    small_edge_mask = edge_widths <= 1e-12
    is_internal_edge = ~np.isin(edges, boundary_nodes).any(axis=1)
    small_edge_mask = small_edge_mask & is_internal_edge

    # removing the internal smal edges 
    if np.any(small_edge_mask):
        edges = edges[~small_edge_mask]
        edge_widths = edge_widths[~small_edge_mask]
        edge_lengths = edge_lengths[~small_edge_mask]
        edge_lines = edge_lines[~small_edge_mask]
        q = q[~small_edge_mask]
        C = C[~small_edge_mask]

    # counting edges per node
    num_nodes = len(nodes)
    node_edge_count = np.zeros(num_nodes, dtype=int)
    for edge in edges:
        node_edge_count[edge[0]] += 1
        node_edge_count[edge[1]] += 1
 
    lonely_nodes = np.where(node_edge_count <= 1)[0]
    lonely_nodes = np.setdiff1d(lonely_nodes, boundary_nodes)  # Exclude boundary nodes

    if len(lonely_nodes) > 0:
        # Remove lonely nodes from nodes, x, and y
        valid_nodes = np.setdiff1d(np.arange(num_nodes), lonely_nodes)
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_nodes)}
        
        # Remove lonely nodes from the node list
        nodes = nodes[valid_nodes]
        x = x[valid_nodes]
        y = y[valid_nodes]
        
        # Update boundary and internal nodes
        boundary_nodes = np.where((x == x.min()) | (x == x.max()))[0]
        internal_nodes = np.setdiff1d(np.arange(len(nodes)), boundary_nodes)
        
        # Remove edges connected to lonely nodes and update edge arrays
        valid_edges = ~np.isin(edges, lonely_nodes).any(axis=1)
        edges = edges[valid_edges]
        edge_widths = edge_widths[valid_edges]
        edge_lengths = edge_lengths[valid_edges]
        edge_lines = edge_lines[valid_edges]
      
        # Reindex the edges array using the new node indices
        edges = np.array([[node_mapping[n1], node_mapping[n2]] for n1, n2 in edges])
       # finding and removing islands 
   
    if len(edges) > 0:
        
        # Create adjacency matrix
        num_nodes = len(nodes)
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        for edge in edges:
            adjacency_matrix[edge[0], edge[1]] = 1
            adjacency_matrix[edge[1], edge[0]] = 1

        # Find connected components
        num_components, labels = connected_components(adjacency_matrix, directed=False)

        if num_components > 1:
            print(f'island detetected at step :{steps}')
      
            plot(q0, 'before island detected', nodes0, edges0)
            plot(q, 'after island detected', nodes, edges)
            
            # Find the largest connected component
            component_sizes = np.bincount(labels)
            largest_component_label = np.argmax(component_sizes)

            # Keep only nodes in the largest connected component
            valid_nodes = np.where(labels == largest_component_label)[0]
            node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_nodes)}
            
            # Remove nodes not in the largest component
            nodes = nodes[valid_nodes]
            x = x[valid_nodes]
            y = y[valid_nodes]
            
            # Update boundary and internal nodes
            boundary_nodes = np.where((x == x.min()) | (x == x.max()))[0]
            internal_nodes = np.setdiff1d(np.arange(len(nodes)), boundary_nodes)
            
            # Remove edges connected to removed nodes
            valid_edges = np.isin(edges, valid_nodes).all(axis=1)
            edges = edges[valid_edges]
            edge_widths = edge_widths[valid_edges]
            edge_lengths = edge_lengths[valid_edges]
            edge_lines = edge_lines[valid_edges]
            
            # Reindex the edges array using the new node indices
            edges = np.array([[node_mapping[n1], node_mapping[n2]] for n1, n2 in edges])
            plot(q, 'removing islands', nodes, edges)
            
            x, y, D, Dn, Db, nodes, edges, edge_widths, edge_lengths, edge_lines, internal_nodes, boundary_nodes, num_nodes = removing_nodes()
            plot(q, 'removing lonley nodes ', nodes, edges)

    num_nodes = len(nodes)
    num_edges = len(edges)
    
    D, Db, Dn = D_matrix( edges,num_nodes, num_edges, boundary_nodes, internal_nodes)
    
    return q, C, x, y, D, Dn, Db, nodes, edges, edge_widths, edge_lengths, edge_lines, internal_nodes, boundary_nodes, num_nodes, num_edges
#%%
    
def plot(q, title, nodes, edges, edge_widths):
    """Plot the network with adjusted visualization for periodic edges."""
    # Normalize flux for color mapping and transparency
    q = abs(q)
    q = q / (sum(q) / len(q))
    norm = q / max(q)
    alpha_high = norm > 0.05
    alpha_low = (norm < 0.05) * 0.3
    alpha = alpha_high + alpha_low

    width_norm = edge_widths / max(edge_widths)
    widths = 2 * width_norm + 0.5

    # Build new edge lines that adjust for periodicity
    new_edge_lines = []
    for (i1, i2) in edges:
        p1 = nodes[i1].copy()
        p2 = nodes[i2].copy()
        # If the vertical distance is larger than half the domain, assume periodic crossing.
        if abs(p1[1] - p2[1]) > Ny/2:
            # Shift the node with the larger y downward by L (so it appears at the bottom ghost location)
            if p1[1] > p2[1]:
                p1[1] -= Ny
            else:
                p2[1] -= Ny
        new_edge_lines.append((p1, p2))
    edge_lines = np.array(new_edge_lines)

    # Create the LineCollection for edges with adjusted coordinates
    lc = mc.LineCollection(edge_lines, cmap='cool', alpha=alpha, linewidths=widths)
    lc.set_array(q)  # use flux values for color mapping

    # Plotting setup
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.add_collection(lc)
    x = nodes[:, 0]
    y = nodes[:, 1]

    # Identify and plot boundary nodes as before (including ghost points)
    left_nodes = np.where((x == 0) )[0]
    right_nodes = np.where((x == Nx) )[0]
    
    top_nodes = np.where((y == Ny) & (x != 0) & (x != Nx))[0]
    # Plot top nodes as well as their ghost image at the bottom
    for i in top_nodes:
        plt.scatter(x[i], Ny, color='black', s=40)
        plt.scatter(x[i], 0, color='black', s=40)
    
    for i in left_nodes:
        plt.scatter(nodes[i][0], nodes[i][1], color='blue', s=20)
        
    for i in right_nodes:
        plt.scatter(nodes[i][0], nodes[i][1], color='red', s=20)

    ax.autoscale()  # adjust axes to fit the graph
    ax.set_aspect('equal')
    plt.colorbar(lc, label="Flux")  # add color legend
    plt.ylim((0, Ny))
    # plt.title(title)
    plt.axis('off')
    plt.show()

def PDF(q, tietel):
    bins_count = int(np.sqrt(len(q)))
    hist_values, bin_edges = np.histogram(abs(q), bins=bins_count, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure()
    plt.scatter(bin_centers, hist_values, marker='o', color='black')
    plt.yscale('log')
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title(tietel)
    plt.show()
    
    
def count (edges):
    edge_count = np.zeros(num_nodes, dtype = int)
    for edge in edges:
        edge_count[edge[0]] += 1
        edge_count[edge[1]] += 1
    return edge_count
#%% updating functions

def solve_qP():
    # Assemble block matrix using np.block
    A_block = np.block([[Abb, Abn],
                        [Anb, Ann]])
    # Assemble RHS vector
    rhs = np.concatenate([q_BC, np.zeros(len(internal_nodes))])
    
    # Solve the full system
    P_full = np.linalg.solve(A_block, rhs)
    
    # Extract pressures
    P = np.zeros(num_nodes)
    P[boundary_nodes] = P_full[:len(boundary_nodes)]  # Boundary pressures
    P[internal_nodes] = P_full[len(boundary_nodes):]   # Internal pressures
    
    # Compute edge fluxes
    q = C @ D @ P
    return q, P

def update_wdiths(q, edge_widths):
    relaxing = np.mean(abs(q)**m/edge_widths**n)
    edge_widths = np.maximum(edge_widths, 1e-12)  # Prevent division by zero
    dr_dt = np.abs(q) ** m / (edge_widths ** n)   - beta*relaxing  # Compute dr/dt
    maxdr_dt = np.max(dr_dt)  # Max dr/dt
    dt =abs( 0.1 / (maxdr_dt) ) # Time step, avoiding division by zero
    # dt = 0.001
    dr = dr_dt * dt  # Compute dr
    new_edge_widths = np.maximum(edge_widths + dr,1e-12 )
    return new_edge_widths , dt

#%% setting up inital matrices and vecotrs 

P_BC = np.zeros(len(boundary_nodes))  # setting up the boundary pressure
P_BC[x[boundary_nodes] == np.min(x)] = pleft  # Left boundary pressure
P_BC[x[boundary_nodes] == x.max()] = pright  # Right boundary pressure

q_BC = np.zeros(len(boundary_nodes)) 
q_BC[x[boundary_nodes] == np.min(x)] = q_left 
q_BC[x[boundary_nodes] == x.max()] = q_right  

C = conductance(edge_widths)
Abb, Abn, Anb, Ann = A_matrices( C)

times = [t]

q,P = solve_qP()
q_array = q.copy()

r0 = np.mean(edge_widths)
q0 = q.copy()
nodes0 = nodes.copy()
edges0 = edges.copy()
widths0 = edge_widths.copy()

qn = D.T @ C @ D @ P
left_nodes = np.where((x == x.min()) )[0] #boundry nodes
right_nodes = np.where((x == x.max()) )[0] 


print(f'starting, Nx,Ny = {Nx}, {Ny}, n = {n}, max_steps = {max_steps}')
print(f'number of starting nodes: {len(nodes)}')
print(f'number of starting edges: {len(edges)}')

print(f'start incoming flux = {np.sum(qn[left_nodes]):.1f}')
print(f'start outgoing flux = {np.sum(qn[right_nodes]):.1f}')
print(f'start flux difference {np.sum(qn[left_nodes]) + np.sum(qn[right_nodes]):.2e}')
print(f'start average abs internal flux = {np.mean(abs(qn[internal_nodes])):.2e} ')

q_list = [q]
edges_list = [edges]
nodes_list = [nodes]
widht_list = [edge_widths]

plt.figure()
plt.title('start edge count')
bins = max(count(edges))- min(count(edges))
plt.hist(count(edges), bins=bins )
plt.show()

# running simulation 
while steps < max_steps :
    edge_widths, dt = update_wdiths(q, edge_widths)
    q, C, x, y, D, Dn, Db, nodes, edges, edge_widths, edge_lengths, edge_lines, internal_nodes, boundary_nodes, num_nodes, num_edges= clean_graph()
   
    C = conductance(edge_widths)
    Abb, Abn, Anb, Ann = A_matrices( C)
    q, P = solve_qP()
 
    t += dt
    steps += 1
    edges_list.append(edges)
    q_list.append(q)
    times.append(t)
    nodes_list.append(nodes)
    widht_list.append(edge_widths)
    
    if (steps %1000) == 0:
        print(f'steps = {steps}, time = {time.time()-start:.2f}')
 
times = np.array(times)


plt.figure()
plt.title('final edge count')
bins = max(count(edges))- min(count(edges))
plt.hist(count(edges), bins=bins )
plt.show()


qn = D.T @ C @ D @ P
left_nodes = np.where((x == x.min()) )[0] #boundry nodes
right_nodes = np.where((x == x.max()) )[0] 
print(f'final incoming flux = {np.sum(qn[left_nodes]):.1f}')
print(f'final outgoing flux = {np.sum(qn[right_nodes]):.1f}')
print(f'final flux difference {np.sum(qn[left_nodes]) + np.sum(qn[right_nodes]):.2e}')
print(f'final average abs internal flux = {np.mean(abs(qn[internal_nodes])):.2e} ')

print(f'number of end nodes: {len(nodes)}')
print(f'number of end edges: {len(edges)}')
#%% plotting final flux and density function
#plotting flow network 
plot(q, f'Final Network, n= {n}, steps ={steps}', nodes, edges, edge_widths)
plot(q0, f'start Network, n= {n}, steps ={steps}', nodes0, edges0, widths0)

# plotting probabilyt function
PDF (q0 ,f"start PDF n = {3} " )
PDF (q ,f"final PDF n = {3} " )

print(f'total time is:{time.time()-start:.2f}')
q_normalized = [abs(q)/ np.mean(abs(q)) for q in q_list]
#%% Animation setup
def periodic_lines (edges, nodes):
    new_edge_lines = []
    for (i1, i2) in edges:
        p1 = nodes[i1].copy()
        p2 = nodes[i2].copy()
        # If the vertical distance is larger than half the domain, assume periodic crossing.
        if abs(p1[1] - p2[1]) > Ny/2:
            # Shift the node with the larger y downward by L (so it appears at the bottom ghost location)
            if p1[1] > p2[1]:
                p1[1] -= Ny
            else:
                p2[1] -= Ny
        new_edge_lines.append((p1, p2))
    edge_lines = np.array(new_edge_lines)
    return edge_lines
    

skip = 5  # Setting up the animation
N_frames = int(len(times) / skip) - 1
q_min, q_max = min(q_normalized[0]), max(q_normalized[0])

# Setting up the plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0-1, Nx+1)
ax.set_ylim(0-1, Ny+1)
ax.set_aspect('equal')

norm = q_normalized[0] / np.max(q_normalized[0])
alpha_high = norm > 0.05
alpha_low = (norm < 0.05) * 0.3
alpha = alpha_high + alpha_low

widths = widht_list[0]
width_norm = widths/max(widths)
widths= 2 * width_norm +0.3

# Initial dummy LineCollection (will be replaced in animation)
edge_lines = periodic_lines(edges, nodes)


lc = mc.LineCollection(edge_lines, cmap='cool', alpha = alpha, linewidths = widths)
lc.set_clim(q_min, q_max)
ax.add_collection(lc)
ax.set_aspect('equal')
plt.ylim((0, Ny))

# Add colorbar
# plt.colorbar(lc, label="Flux")

# Scatter plot for boundary nodes

left_nodes = np.where((x == 0) )[0]
right_nodes = np.where((x == Nx) )[0]
   

for i in left_nodes:
       plt.scatter(nodes[i][0], nodes[i][1], color='blue', s=20)
       
for i in right_nodes:
       plt.scatter(nodes[i][0], nodes[i][1], color='red', s=20)
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)
top_scat = ax.scatter([], [], color='black', s=40)
bottom_scat = ax.scatter([], [], color='black', s=40)

plt.title("Mean Abs Flux")
plt.axis("off")
plt.show()


def animate(i):
    """Update function for animation."""
    global lc, top_scat, bottom_scat
    t = int(i * skip) # updating the step
    t = np.minimum(t, len(q_list)) # preventing out of range
    
    # getting arrays for step t form their lists
    q = q_normalized[t]
    edges = edges_list[t]
    nodes = nodes_list[t]
    x = nodes[:, 0]
    y = nodes[:, 1]
    widths = widht_list[t]
    width_norm = widths/max(widths)
    widths= 2 * width_norm +0.5
    # Normalize color range
    q_min, q_max = np.min(q), np.max(q)
  
    edge_lines = periodic_lines(edges, nodes)
    top_nodes = np.where( (y == Ny) & (x != 0) & (x != Nx))[0]
    
    Nt = len (top_nodes)
    
    if len(top_nodes)> 0:
        top_positions = np.column_stack((x[top_nodes], np.ones(Nt)*Ny) )
        bottom_positions = np.column_stack((x[top_nodes], np.zeros(Nt))) 
        
    else :
       top_positions =  np.empty((0, 2))
       bottom_positions =np.empty((0, 2))
        
    top_scat.set_offsets(top_positions)
    bottom_scat.set_offsets(bottom_positions)
     
    # Normalize flux values and set alpha
    norm = q / np.max(q)
    
    
    alpha_high = norm > 0.05
    alpha_low = (norm < 0.05) * 0.3
    alpha = alpha_high + alpha_low
   
    # Clear all collections (including previous LineCollection)
    for coll in ax.collections:
       if isinstance(coll, mc.LineCollection):
           coll.remove()

    # Create a new LineCollection with updated edges
    lc = mc.LineCollection(edge_lines, cmap='cool', alpha=alpha, linewidths = widths)
    lc.set_array(q)
    lc.set_clim(q_min, q_max)

    # Add new edges to plot
    ax.add_collection(lc)
    # Update time text
    time_text.set_text(f'Steps: {t}')
    plt.ylim((0, Ny))

    return lc, time_text

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=N_frames, interval=500)

# # Save the animation
# anim.save('per_removing.gif', writer=animation.PillowWriter(fps=4))