
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import time 
from matplotlib import animation
import matplotlib.collections as mc
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

#%% inital set up parameters

plt.close('all')
pi = np.pi
start = time.time() #starting time 
np.random.seed(6)  # Setting a seed to reproduce results

# network values
Nx, Ny = 20,20  
Nb = 15# edge nodes left and right
num_nodes = Nx * Ny    # Total nodes

#flow values
I0= 0.2
gamma = 1.15
beta = 0.5/10
mu =1

# conditions time and steps
t0 = 0
tmax = 100
dt = 0.01
t = t0
max_steps = 3000
steps = 1

# the boundrys of the intrernal nodes 
left = 1
right = Nx-1
top = Ny -1
botom = 1

#%% importing stations
stations_file = "stations.csv"
stations_df = pd.read_csv(stations_file, encoding="utf-8")

# Filter for Dutch stations
dutch_stations_df = stations_df[stations_df["Land"] == "Netherlands"].copy()

# Apply Utrecht mask before extracting values
utrecht_stations_df = dutch_stations_df[
     (dutch_stations_df["Lat"] > 51.90) & 
     (dutch_stations_df["Lat"] < 52.29) & 
     (dutch_stations_df["Lon"] > 4.865) & 
     (dutch_stations_df["Lon"] < 5.56)
 ].copy()  # Ensure we're working with a separate copy

# Extract station positions and names
station_positions_np = np.array(utrecht_stations_df[["Lat", "Lon"]])
filtered_stations = utrecht_stations_df["Station"].values  # Correct filtering

lat = station_positions_np[:, 0]
lon = station_positions_np[:, 1]

# Min-Max Scaling function
def min_max_scale(values, new_min=3, new_max=Ny-3):
     old_min, old_max = np.min(values), np.max(values)
     return new_min + (values - old_min) * (new_max - new_min) / (old_max - old_min)

# Scale to grid (factor of grid to prevent boundry effects)
scaled_x = min_max_scale(lon, 3, Nx-3)
scaled_y = min_max_scale(lat, 3, Ny-3)
scaled_positions = np.column_stack((scaled_x, scaled_y))

stations = len(lon)

#%% setting up the network 

x = np.random.uniform(left, right, num_nodes)
y = np.random.uniform(botom , top , num_nodes)

nodes = np.column_stack((x, y))
nodes = np.concatenate((nodes, scaled_positions))

nodes = scaled_positions # 
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

num_nodes = len(nodes)
num_edges = len(edges)

#setting up edge lenghts, edge widhts , edge lines
edge_lengths = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
edge_widths = np.ones(num_edges) 
edge_lines = np.array([(nodes[i], nodes[j]) for i, j in edges])

station_nodes = np.arange(len(nodes) - len(scaled_positions), len(nodes))

boundary_nodes = np.random.choice(station_nodes, size=2, replace=False)
internal_nodes = np.setdiff1d(np.arange(num_nodes), boundary_nodes) # index internal nodes

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

def conductance(edge_widths):
    C =  np.diag((pi*edge_widths**4)/(8* mu* edge_lengths))
    return C

def A_matrices( C):   
    Abb = Db.T @ C @ Db
    Abn = Db.T @ C @ Dn
    Anb = Dn.T @ C @ Db
    Ann = Dn.T @ C @ Dn
    return Abb, Abn, Anb, Ann

#%% plot function

def plot(q, title, nodes, edges, edge_widths, boundary_nodes, filename= None):
    """Plot the network with adjusted visualization for periodic edges."""
    # Normalize flux for color mapping and transparency
    q = abs(q)
    q = q / sum(q) 
    norm = q / max(q)
    alpha_high = norm > 0.05
    alpha_low = (norm < 0.05) * 0.3
    alpha = alpha_high + alpha_low

    width_norm = edge_widths / max(edge_widths)
    widths = 2 * width_norm + 0.3

    # Build new edge lines that adjust for periodicity
    edge_lines = np.array([(nodes[i], nodes[j]) for i, j in edges])
  
    # Create the LineCollection for edges with adjusted coordinates
    lc = mc.LineCollection(edge_lines, cmap='cool', alpha=alpha, linewidths=widths)
    lc.set_array(q)  # use flux values for color mapping

    # Plotting setup
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.add_collection(lc)

    for i in range(len(station_nodes)):
       plt.scatter(nodes[station_nodes[i]][0], nodes[station_nodes[i]][1], color='black', s=20, zorder = 5)
       
    for i in boundary_nodes:
        plt.scatter(nodes[i][0], nodes[i][1], color='red', s=50, zorder = 5)

    ax.autoscale()  # adjust axes to fit the graph
    ax.set_aspect('equal')
    plt.axis('off')  # Hide both axes
    plt.title(title)
    plt.show()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)  
    
#%% updating functions

def solve_qP():
    "boundary flux"
    # Assemble block matrix using np.block
    A_block = np.block([[Abb, Abn],[Anb, Ann]])
    
    # Assemble RHS vector
    rhs = np.concatenate([q_BC, np.zeros(len(internal_nodes))])
    
    try: #try cholonsky decomposition for speed
        A_block += 1e-10 * np.eye(A_block.shape[0]) # add eye matrix for stability
        c, low = cho_factor(A_block)
        P_full = cho_solve((c, low), rhs)
       
    except: # fall back on psuedo inverse for stabilyt
        P_full = np.dot(np.linalg.pinv(A_block), rhs)
        print("fall back on psuedo inverse")
        
    # Extract pressures
    P = np.zeros(num_nodes)
    P[boundary_nodes] = P_full[:len(boundary_nodes)]  # Boundary pressures
    P[internal_nodes] = P_full[len(boundary_nodes):]   # Internal pressures
    
    # Compute edge fluxes
    q = C @ D @ P
    return q, P

def update_wdiths(q, edge_widths):
    edge_widths = np.maximum(edge_widths, 1e-12)  # Prevent division by zero
    D = pi* edge_widths**4/(8*mu)
    dD_dt = np.abs(q)**gamma/(np.abs(q)**gamma +1)  -beta* D 
    dr_dt = 2*mu/(pi* edge_widths**3) *dD_dt
    dt =1
    dr = dr_dt * dt  # Compute dr
    new_edge_widths = np.maximum(edge_widths + dr,1e-12 )
    return new_edge_widths , dt

#%% setting up inital matrices and vecotrs 
q_BC = np.zeros(len(boundary_nodes)) 
q_BC[0] = I0
q_BC[1] = -I0

C = conductance(edge_widths)

Abb, Abn, Anb, Ann = A_matrices( C)
times = [t]
q, P = solve_qP()

r0 = np.mean(edge_widths)
q_array = q.copy()
q0 = q.copy()
nodes0 = nodes.copy()
edges0 = edges.copy()
widths0 = edge_widths.copy()

q_list = [q]
widht_list = [edge_widths]
boundary_list = [boundary_nodes]

# %%running simulation 
while steps < max_steps :
   
    #picking boundary nodes and internal nodes
    boundary_nodes = np.random.choice(station_nodes, size=2, replace=False)
    internal_nodes = np.setdiff1d(np.arange(num_nodes), boundary_nodes) # index internal nodes
   
    #finding the flow and updating edge witdths 
    D, Db, Dn  = D_matrix(edges, num_nodes, num_edges, boundary_nodes, internal_nodes)
    C = conductance(edge_widths)
    Abb, Abn, Anb, Ann = A_matrices( C)
    q, P = solve_qP()
    edge_widths, dt = update_wdiths(q, edge_widths)
 
    #updating evertying 
    t += dt
    steps += 1
    q_list.append(q)
    times.append(t)
    widht_list.append(edge_widths)
    boundary_list.append(boundary_nodes)
    
    if (steps %1000) == 0:
        print(f'steps = {steps}, time = {time.time()-start:.2f}')
  
times = np.array(times)

qn = D.T @ C @ D @ P

#printing the internal node flow as a sanity check
print(f'mean abs internal node flux : {np.mean(abs(qn[internal_nodes]))}')

#%% plotting final flux and density function
#plotting flow network 
plot(q, f'Final Network, gamma = {gamma}, steps ={steps}', nodes, edges, edge_widths, boundary_list[-1])
plot(q0, f'start Network, gamma = {gamma}', nodes0, edges0, widths0, boundary_list[0])

print(f'total time is:{time.time()-start:.2f}')
#%% plotting a step a for thesisi

step = max_steps
plot(q_list[step-1], ' ', nodes, edges, widht_list[step-1], boundary_list[step-1])
plot(q_list[step-1], ' ', nodes, edges, widht_list[step-1], boundary_list[step-1], 'final_flow_few115')
step =1
plot(q_list[step-1], ' ', nodes, edges, widht_list[step-1], boundary_list[step-1])
#%% finding the network of edge of a certing cutoff size

def network (step, cutof, filename = None):
    edge_widths = widht_list[step-1]
    mask = edge_widths >= cutof
    maskedges = edges[mask]

    # Convert edge indices to actual coordinates
    edge_lines = np.array([(nodes[i], nodes[j]) for i, j in maskedges])
    lc = mc.LineCollection(edge_lines, color="darkgrey", linewidths=1, zorder=1)

    # Plotting setup
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.add_collection(lc)
    plt.axis('off')
    # plt.title(step)

    # Plot station nodes
    for i, node in enumerate(station_nodes):
        x, y = nodes[node]  # Get actual node positions
        plt.scatter(x, y, color="black", s=20, zorder = 2)
        plt.text(x + 0.1, y + 0.1, filtered_stations[i], fontsize=5, alpha=0.75, zorder = 3)


    ax.autoscale()  # Adjust axes to fit the graph
    ax.set_aspect("equal")
    plt.show()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory
  
network(max_steps, 0.55)

#%%  animating
q_normalized = [abs(q)/ (np.mean(abs(q))+ 0.001) for q in q_list]

skip = 10  #skipping simulated frames when animating

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


edge_lines = np.array([(nodes[i], nodes[j]) for i, j in edges])

lc = mc.LineCollection(edge_lines, cmap='cool', alpha = alpha, linewidths = widths)
lc.set_clim(q_min, q_max)
ax.add_collection(lc)
ax.set_aspect('equal')
plt.ylim((0, Ny))
plt.tight_layout()

# Scatter plot for boundary nodes
for i in range(len(station_nodes)):
   plt.scatter(nodes[station_nodes[i]][0], nodes[station_nodes[i]][1], color='black', s=20)


# Text for time steps
time_text = ax.text(0.11, 0.95, '', transform=ax.transAxes, fontsize=12)
bound_scat =  ax.scatter([], [], color='red', s=50)

# plt.title("Mean Abs Flux")
plt.axis('off')
plt.show()

def animate(i):
    """Update function for animation."""
    t = int(i * skip)  # Get the step
    t = np.minimum(t, len(q_list) - 1)  # Prevent out-of-range access

    # Get updated values for step `t`
    q = q_normalized[t]
    widths = widht_list[t]

    # Normalize edge widths
    width_norm = widths / max(widths)
    plot_widths = 2 * width_norm + 0.3

    # Normalize alpha transparency
    norm = q / (np.max(q)+ 0.001)
    alpha_high = norm > 0.05
    alpha_low = (norm < 0.05) * 0.3
    alpha = alpha_high + alpha_low

    # Update `lc` instead of recreating it
    lc.set_segments(edge_lines)  # Update edges
    lc.set_array(q)  # Update colors
    lc.set_linewidths(plot_widths)  # Update widths
    lc.set_alpha(alpha)  # Update transparency

    # Update boundary nodes
    boundary_nodes = boundary_list[t]
    bound_pos = np.column_stack((nodes[boundary_nodes, 0], nodes[boundary_nodes, 1]))
    bound_scat.set_offsets(bound_pos)

    # Update time text
    time_text.set_text(f'Steps: {t}')

    return lc, bound_scat, time_text

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=N_frames, interval=500)

# # # Save the animation
# anim.save('utrecht.gif', dpi = 100 , writer=animation.PillowWriter(fps=2), savefig_kwargs={'bbox_inches': 'tight'})
