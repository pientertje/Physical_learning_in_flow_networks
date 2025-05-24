
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import time 
from matplotlib import animation
import matplotlib.collections as mc
from scipy.linalg import cho_factor, cho_solve

plt.close('all')
pi = np.pi
start = time.time() #starting time 
np.random.seed(6)  # Setting a seed to reproduce results
#%% inital conditions network
Nx, Ny = 15, 15 # network size
num_nodes = Nx * Ny  # Total nodes
#%% initial conditions simulation
pleft = 0 # Initial pressure
pright = 10

mu = 1 # conductunce constant

max_steps = 3000 # max steps
t = 0 # intial time and step
steps = 0

# the boundrys of the intrernal nodes 
left = 1
right = Nx-1
top = Ny -1/2
botom = 1/2

lam = 50 # Chemical release rate
xi = 100 # Conductance adjustment rate
tau = 40  # Update interval for conductance changes

#initial condition of target pressure
b = 2/max_steps
B = 2
omega = 2*pi/max_steps
#%% setting up the network 

# generating randomly distributed points
x = np.random.uniform(left, right, num_nodes)
y = np.random.uniform(botom , top , num_nodes)

# adding target nodes
x = np.concatenate([x, [0.5 * Nx], [0.5* Nx]])
y = np.concatenate([y, [1/4 * Ny], [3/4 *Ny]])

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

#finding internal and boundary nodes ( input and outpout )
input_nodes = np.where((x == x.min()) | (x == x.max()))[0] #boundry nodes
internal_nodes = np.setdiff1d(np.arange(num_nodes), input_nodes) # index internal nodes
output_nodes = np.where(x == 0.5 * Nx)[0]

#setting up edge lenghts, edge widhts , edge lines
edge_lengths = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
edge_widths = np.random.uniform(5, 14, num_edges) # generating widht of the edges
edge_lines = np.array([(nodes[i], nodes[j]) for i, j in edges])

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

def conductance(edge_widths):
    C =  np.diag((pi*edge_widths**4)/(8* mu* edge_lengths))
    return C

def A_matrices( C, Db, Dn):   
    Abb = Db.T @ C @ Db
    Abn = Db.T @ C @ Dn
    Anb = Dn.T @ C @ Db
    Ann = Dn.T @ C @ Dn   
    return  Abb, Abn, Anb, Ann

#%% plotting/saving function
    
def plot(q, title, nodes, edges, edge_widths, filename= None):
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

    # Create the LineCollection for edges with adjusted coordinates
    lc = mc.LineCollection(edge_lines, cmap='cool', alpha=alpha, linewidths=widths)
    lc.set_array(q)  # use flux values for color mapping

    # Plotting setup
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.add_collection(lc)

    for i in input_nodes:
        plt.scatter(nodes[i][0], nodes[i][1], color='blue', s=40, zorder = 5)
    
    for i in output_nodes:
        plt.scatter(nodes[i][0], nodes[i][1], color='red', s=40, zorder = 5)

    ax.autoscale()  # adjust axes to fit the graph
    ax.set_aspect('equal')
    # plt.colorbar(lc, label="Flux")  # add color legend
    plt.axis('off')
    plt.ylim((0, Ny))
    plt.title(title)
    plt.show()
    
    if filename: # save if wanted
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)  

#%% updating functions

def solve_qP():
    P = np.zeros(num_nodes)
    P[input_nodes] = P_BC

    try:
        c, low = cho_factor(Ann) #try cholesky decomposition
        P_internal = cho_solve((c, low), -Anb @ P_BC)
        
    except:
        P_internal = np.linalg.pinv(Ann) @ (-Anb @ P_BC) # fall back to pseudo  
        
    P[internal_nodes] = P_internal
    q = np.diag(C) * (D @ P) 
    return q, P

#%% setting up inital matrices and vecotrs 

P_BC = np.array([pleft, pright])
D, Db, Dn = D_matrix( edges,num_nodes, num_edges, input_nodes, internal_nodes)
C = conductance(edge_widths)
Abb, Abn, Anb, Ann = A_matrices( C, Db, Dn)

times = [t]

q,P = solve_qP()
q_array = q.copy()
qn = D.T @ C @ D @ P

print(f'start mean abs internal node flux : {np.mean(abs(qn[internal_nodes]))}')
print(f'starting, Nx,Ny = {Nx}, {Ny},  max_steps = {max_steps}')

#%%

s_plus = np.zeros(num_nodes)  # Chemical plus at nodes
s_minus = np.zeros(num_nodes)  # Chemical minus at nodes
S_plus = np.zeros(len(edges))  # Chemical plus in edges
S_minus = np.zeros(len(edges))  # Chemical minus in edges


def s_initial():
    """Initialize chemicals at output nodes based on pressure error."""
    for i, a in  enumerate(output_nodes):
        error = P[a] - des_P[i]
        if error > 0:
            s_plus[a] = lam * (P[a] - des_P[i])
        elif error < 0:
            s_minus[a] = lam * (des_P[i] - P[a])

def update_s():
    """"""
    global s_plus, s_minus, S_plus, S_minus
    S_plus.fill(0)
    S_minus.fill(0)
    
    n1 = edges[:, 0]
    n2 = edges[:, 1]
    q_abs = np.abs(q)
    source = np.where(q > 0, n1, n2)
    target = np.where(q > 0, n2, n1)
    
    # Total outgoing current per source node
    total_current = np.bincount(source, weights=q_abs, minlength=num_nodes)
    
    #fraction of current per edge
    valid = total_current[source] > 0
    frac = np.zeros_like(q)
    frac[valid] = q_abs[valid] / total_current[source][valid]
    
    # Distribute s_plus and s_minus from source nodes to edges
    S_plus[:] = frac * s_plus[source]
    S_minus[:] = frac * s_minus[source]
    
    # distiribute chemical from edges into nodes 
    s_plus[:] = np.bincount(target, weights=S_plus, minlength=num_nodes)
    s_minus[:] = np.bincount(target, weights=S_minus, minlength=num_nodes)
    
    return S_plus, S_minus
        
#%%

a = P[output_nodes] # start target pressrue

def linear (x, a, b):
    #linear function used for linear target pressure
    return a + b * x

def sinus( x, a , b, omega):
    #sinus function used for sinus target pressure
    return a + b * np.sin(omega * x)

des_P = sinus ( steps, a, B, omega)

#initializing arrays
s_initial()
S_plus_list = np.zeros((max_steps, num_edges))
S_minus_list = np.zeros((max_steps, num_edges))
q_list = np.zeros((max_steps, num_edges))
widht_list = np.zeros((max_steps, num_edges))
Plist =np.zeros((max_steps, len(output_nodes)))

#initialize release times
release_time = []
releases = 0
previous_release_step = 0

while steps < max_steps:
    
    total_chemical = np.sum(s_plus) + np.sum(s_minus)
   
    if total_chemical == 0:
        P_BC  *=  -1 #fliping boundary pressures
        
        #selecting a linear, sinus or constant des pressure
        des_P = linear ( steps, a, b,)
        # des_P = sinus ( steps, a, B, omega)
        # des_P = np.array([3,7])
      
        if np.any(P_BC < 0):
            des_P *= -1
       
        #finding q and P
        C = conductance(edge_widths)
        D, Db, Dn = D_matrix(edges, num_nodes, num_edges, input_nodes, internal_nodes)
        Anb = Dn.T @ C @ Db
        Ann = Dn.T @ C @ Dn   
        q, P = solve_qP()
       
        # releasing chemical
        s_initial() 
        releases += 1
        release_time.append(steps - previous_release_step)
        previous_release_step = steps
   
    # getting chemical flow
    S_plus, S_minus = update_s()  

    # updating edge widht based on chemical flow 
    dC_dt = xi * (S_plus - S_minus)  # Changed to use edge_S values
    dr_dt = 2 * edge_lengths/pi * 1/(edge_widths**3) * dC_dt
    dt =1
    dr = dr_dt * dt
    edge_widths = np.maximum(edge_widths + dr, 1e-12)
    
    C = conductance(edge_widths)
    Anb = Dn.T @ C @ Db
    Ann = Dn.T @ C @ Dn   
    q, P = solve_qP() 
    
    # appanding usfull values
    S_plus_list[steps] = S_plus
    S_minus_list[steps] = S_minus
    q_list[steps]= q
    widht_list[steps] = edge_widths 
    Plist[steps] = P[output_nodes]
    
    # updating step
    steps += 1
 
S_total_list = S_plus_list - S_minus_list

qn = D.T @ C @ D @ P
print(f'end mean abs internal node flux : {np.mean(abs(qn[internal_nodes]))}')

#%% printing and plotting chemical release times
print(f'relases: total {len(release_time )}, mean = {np.mean(release_time[2:]):.1f}, max = {np.max(release_time)}, min = {np.min(release_time[2:])}')

counts, bin_edges = np.histogram(release_time[2:], bins=10)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
#plot
plt.figure(figsize=(10, 6))
plt.bar(bin_centers, counts, width=bin_edges[1]-bin_edges[0], 
            edgecolor='k', alpha=0.7)
# Add average line and text
avg_interval = np.mean(release_time[2:])
plt.axvline(avg_interval, color='r', linestyle='--', 
                label=f'Average: {avg_interval:.1f} steps')   
plt.xlabel('Time between releases (steps)')
plt.ylabel('Count')
plt.title('Distribution of Release Intervals')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%%
#extract presuure at each output node
P1 = np.abs(Plist[::, 0])
P2 = np.abs( Plist[::,1])


#set deseried pressure acording to used pressure
# desP1 = sinus(np.arange(0, len(P1)), a[0], B, omega)
# desP2 = sinus(np.arange(0, len(P2)), a[1], B, omega)

# desP1 = np.ones(len(P1)) * 3
# desP2 = np.ones(len(P2)) * 7

desP1 = linear(np.arange(0, len(P1)), a[0], b)
desP2 = linear(np.arange(0, len(P2)), a[1], b)

error1 = np.abs (P1 - desP1)
error2 = np.abs(P2 - desP2)

#making ther error kwadratic
# error1 = (P1 - desP1)**2
# error2 = (P2 - desP2)**2


plt.figure()
# plt.title('pressure')
plt.plot(np.arange(0, len(P1),1), P1, color = 'k')
plt.plot(np.arange(0, len(P2),1), P2, color = 'b')
plt.plot(np.arange(0, len(P1)), desP1 , "k--")
plt.plot(np.arange(0, len(P2)), desP2, "b--")
plt.xlim((0, steps))
plt.xlabel('step')
plt.ylabel('Pressure')
# plt.savefig('constant_pres', dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
# plt.title(' log error')
plt.plot(np.arange(0, len(P1),1), error1, color = 'k')
plt.plot(np.arange(0, len(P2),1), error2, color = 'b')
plt.xlim((0 , len(P1)))
plt.yscale('log')
plt.xlabel('step')
plt.ylabel('error')
# plt.savefig('chemical_error', dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
# plt.title('error')
plt.plot(np.arange(0, len(P1),1), error1, color = 'k')
plt.plot(np.arange(0, len(P2),1), error2, color = 'b')
plt.xlim((-0.002* len(P1) , len(P1)))
plt.ylim(-0.01, 0.1 +min(np.max(error1), np.max(error2)))
plt.xlabel('step')
plt.ylabel('error')
# plt.savefig('chemical_log_error', dpi=300, bbox_inches='tight')
plt.show()


#%% plotting final flux and density function

plot(q, '', nodes, edges, edge_widths) #plotting end flow network 
plot(q_list[0], '', nodes, edges, widht_list[0], 'chemcial_flux') #plotting start flow network 

print(f'total time is:{time.time()-start:.2f}, releases = {releases}')

#%% animate chemical advection

S_total_list = S_plus_list + S_minus_list
S_total = S_total_list > 0

widht_array = np.array(widht_list)
width_norm = widht_array/np.max(widht_array, axis = 0)
anim_widths = 2 * width_norm + 0.3

#%% chemical animation
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'  # For Apple Silicon Macs

skip = 1  # Setting up the animation
N_frames = int(len(P1)/skip) - 1
N_frames = 334

aspect = Nx / Ny  # Data aspect ratio
fig_height = 8    # Arbitrary height in inches
fig_width = fig_height * aspect  # Width adjusted to match data aspect
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

widths = widht_list[0]
width_norm = widths/max(widths)
widths= 2 * width_norm +0.3

binary_cmap = ListedColormap(['cyan', 'Magenta'])
lc = mc.LineCollection(edge_lines, cmap= binary_cmap, alpha = 1, linewidths = widths)
ax.add_collection(lc)
ax.set_aspect('equal')
plt.ylim((0, Ny))

#plot input and output nodes
for i in range(len(output_nodes)):
   plt.scatter(nodes[output_nodes[i]][0], nodes[output_nodes[i]][1], color='red', s=40, zorder =5)
   
for i in range(len(input_nodes)):
    plt.scatter(nodes[input_nodes[i]][0], nodes[input_nodes[i]][1], color='blue', s=40, zorder = 5)
# Text for time steps
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)
plt.axis('off')
plt.show()

def animate(i):
    """Update function for animation."""
    global lc, time_text  # Removed top_scat, bottom_scat as they aren't used
    
    # Calculate the time step
    t = int(i * skip)
    
    # Get arrays for step t from their lists
    S =  np.abs(S_total[t])
    widths = anim_widths[t]

    # Create a new LineCollection with the pre-defined colormap
    lc = mc.LineCollection(edge_lines, cmap= binary_cmap, linewidths=widths)
    
    lc.set_array(S)
    
    # Add the LineCollection to the plot
    ax.add_collection(lc)
    
    # Update time text
    time_text.set_text(f'Steps: {t}')
    ax.set_ylim(0, Ny)  # Use ax.set_ylim instead of plt.ylim for consistency
    
    return lc, time_text

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=N_frames, interval=25)

# anim.save(
#     'chemical_advection.mp4',
#     writer='ffmpeg',
#     fps=6,
#     dpi=150
# )
#%% flux flow animation
skip = 3  # Setting up the animation
N_frames = int(len(P1)/skip) - 1
N_frames = 334
q_normalized = [abs(q)/ (np.mean(abs(q))+ 0.001) for q in q_list]

import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'  # For Apple Silicon Macs

aspect = Nx / Ny  # Data aspect ratio
fig_height = 8    # Arbitrary height in inches
fig_width = fig_height * aspect  # Width adjusted to match data aspect
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

widths = widht_list[0]
width_norm = widths/max(widths)
widths= 2 * width_norm +0.3

lc = mc.LineCollection(edge_lines, cmap= 'cool', alpha = 1, linewidths = widths)
ax.add_collection(lc)
ax.set_aspect('equal')
plt.ylim((0, Ny))

#plot input and output nodes
for i in range(len(output_nodes)):
   plt.scatter(nodes[output_nodes[i]][0], nodes[output_nodes[i]][1], color='red', s=40, zorder =5)
   
for i in range(len(input_nodes)):
    plt.scatter(nodes[input_nodes[i]][0], nodes[input_nodes[i]][1], color='blue', s=40, zorder = 5)
# Text for time steps
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)
plt.axis('off')
plt.show()

def animate(i):
    """Update function for animation."""
    global lc, time_text  # Removed top_scat, bottom_scat as they aren't used
    
    # Calculate the time step
    t = int(i * skip)
    
    # Get arrays for step t from their lists
    q = q_normalized[t]

    
    norm = q / (np.max(q)+ 0.001)
    alpha_high = norm > 0.05
    alpha_low = (norm < 0.05) * 0.3
    alpha = alpha_high + alpha_low

    # Create a new LineCollection with the pre-defined colormap
    lc.set_array(q_normalized[t])
    lc.set_linewidths(anim_widths[t])
    lc.set_alpha(alpha)
    
    # Update time text
    time_text.set_text(f'Steps: {t}')
    ax.set_ylim(0, Ny)  # Use ax.set_ylim instead of plt.ylim for consistency
    
    return lc, time_text

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=N_frames, interval=25)
