
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import time 
from matplotlib import animation
import matplotlib.collections as mc
from scipy.linalg import cho_factor, cho_solve
import matplotlib.ticker as mticker
plt.close('all')
pi = np.pi
start = time.time() #starting time 
np.random.seed(6)  # Setting a seed to reproduce results

#%% importing iris dataset

from ucimlrepo import fetch_ucirepo 
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

#%%

#find all iris types
flower_types = np.unique(y)
print("Found flower types:", flower_types)

# Create a simple mapping: first type = 0, second type = 1, third type = 2
y_numeric = np.zeros(len(y), dtype=int)
for i, label in enumerate(y):
    if label == flower_types[0]:
        y_numeric[i] = 0
    elif label == flower_types[1]:
        y_numeric[i] = 1
    elif label == flower_types[2]:
        y_numeric[i] = 2
y = y_numeric

num_samples = len(X)

random_order = np.arange(num_samples)  
np.random.shuffle(random_order)  # Mix it up randomly

# Rearrange both X and y using the same random order
X = X[random_order]
y = y[random_order]

# Normalize features to range [5, 9]
for i in range(4):
    Xi = X[:, i]
    Xi_min = np.min(Xi)
    Xi_max = np.max(Xi)
    Xi_norm = 4 * (Xi - Xi_min) / (Xi_max - Xi_min)
    X[:, i] = 5 + Xi_norm

# Find indices for each class in the shuffled data
iris_0_indices = np.where(y == 0)[0]
iris_1_indices = np.where(y == 1)[0]
iris_2_indices = np.where(y == 2)[0]

# Randomly select 25 samples from each class for training
training_0 = np.random.choice(iris_0_indices, size=25, replace=False)
training_1 = np.random.choice(iris_1_indices, size=25, replace=False)
training_2 = np.random.choice(iris_2_indices, size=25, replace=False)

# Combine training indices
training = np.concatenate([training_0, training_1, training_2])

# Get testing indices (remaining samples)
testing = np.setdiff1d(np.arange(len(X)), training)

# Split the data
X_training = X[training]
X_testing = X[testing]
y_training = y[training]
y_testing = y[testing]

# Shuffle the training data
training_indices = np.arange(len(X_training))
np.random.shuffle(training_indices)
X_training = X_training[training_indices]
y_training = y_training[training_indices]

len_data = len(X_training)
training_steps = 750

#%% inital conditions network
Nx, Ny = 12, 12 # Example values
num_nodes = Nx * Ny  # Total nodes

pleft = 0 # Initial pressure
pright = 5

mu = 1 # conductunce constant

max_steps = len_data * training_steps # steps in simulation
t = 0 # intial time and step
steps = 0

# the boundrys of the intrernal nodes 
left = 1
right = Nx-1

top = Ny 
botom =0

lam = 100 # Chemical release rate
xi = 50# Conductance adjustment rate
tau = 40  # Update interval for conductance changes

#initial condition of target pressure
b = 1/max_steps
B = 2
omega = 2*pi/max_steps
#%% setting up the network 

# generating randomly distributed points
x = np.random.uniform(left, right, num_nodes)
y = np.random.uniform(botom , top , num_nodes)

# adding output nodes
x = np.concatenate([x, [0.5 * Nx], [0.5* Nx], [0.5 *Nx]] )
y = np.concatenate([y, [2.5/10 * Ny], [5/10 *Ny], [7.5/10 *Ny] ])



# adding left input nodes
x = np.concatenate([x, [0], [0], [0], [0]] )
y = np.concatenate([y, [1/5*Ny], [2/5*Ny], [3/5*Ny], [4/5*Ny] ])

# adding right input nodes
x = np.concatenate([x, [Nx], [Nx], [Nx], [Nx] ])
y = np.concatenate([y, [1/5*Ny], [2/5*Ny], [3/5*Ny], [4/5*Ny] ])

nodes = np.column_stack((x, y))
num_nodes = len(nodes)

tri = scipy.spatial.Delaunay(nodes)  # Delaunay triangulation

# Extract edges from the triangulation
simplices = tri.simplices  # nodes of triangels 
edges = np.vstack([
    simplices[:, [0, 1]],
    simplices[:, [1, 2]],
    simplices[:, [2, 0]]]) 

# Remove duplicate edges and sort
edges = np.unique(np.sort(edges, axis=1), axis=0)

#finding input- and interal nodes and output nodes (part of internal)
left_nodes   = np.where(x == x.min())[0]
right_nodes  = np.where(x == x.max())[0]
input_nodes = np.concatenate([left_nodes, right_nodes])
internal_nodes = np.setdiff1d(np.arange(num_nodes), input_nodes) # index internal nodes
output_nodes = np.where(x == 0.5 * Nx)[0]

#removing edges conecting the boundary nodes
edges_to_remove = [] 
for n1 in input_nodes: #finding edges to remove
    for n2 in input_nodes:
        edges_to_remove.append([n1, n2])
        edges_to_remove.append([n2, n1])
       
for edge in edges_to_remove: #removing edges
    edge_index = np.where((edges == edge).all(axis=1))[0]
    if edge_index.size > 0:
        edges = np.delete(edges, edge_index, axis=0)

num_edges = len(edges)

#setting up edge lenghts, edge widhts , edge lines
edge_lengths = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
edge_widths = np.random.uniform(5, 14, num_edges) # generating widht of the edges
edge_lines = np.array([(nodes[i], nodes[j]) for i, j in edges])
#%%  Extract submatrices functions

def D_matrix( edges,num_nodes, num_edges, boundary_nodes, internal_nodes):
    """function that generates the D , Db, Dn matrices with edges and nodes
    as input """
    D = np.zeros((num_edges, num_nodes))
    for i, (n1, n2) in enumerate(edges):
        D[i, n1] = 1
        D[i, n2] = -1
    Db = D[:, boundary_nodes]
    Dn = D[:, internal_nodes]
    return D, Db, Dn
    
D, Db, Dn = D_matrix( edges,num_nodes, num_edges, input_nodes, internal_nodes)

def conductance(edge_widths):
    """function that generates the  C matrix using the  edge widths """
    C =  np.diag((pi*edge_widths**4)/(8* mu* edge_lengths))
    return C

def A_matrices( C, Db, Dn):  
    """function that generates the A matricises matrix  using C and D """
    Abb = Db.T @ C @ Db
    Abn = Db.T @ C @ Dn
    Anb = Dn.T @ C @ Db
    Ann = Dn.T @ C @ Dn   
    return  Abb, Abn, Anb, Ann

#%% plot functino
    
def plot(q, title, nodes, edges, edge_widths, filename= None):
    """Plot the network with opional kwarg to save the plot """
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
    lc = mc.LineCollection(edge_lines, cmap='cool',
                           alpha=alpha, linewidths=widths)
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

#%% flow and pressure solver 

def solve_qP():
    """no input, gives flow (q) and pressure (P) array as output """
    P = np.zeros(num_nodes)
    P[input_nodes] = P_BC
    try:
        c, low = cho_factor(Ann)
        P_internal = cho_solve((c, low), -Anb @ P_BC)
    except np.linalg.LinAlgError:
        print("Warning: Ann is ill-conditioned, falling back to pinv")
        P_internal = np.linalg.pinv(Ann) @ (-Anb @ P_BC)
    P[internal_nodes] = P_internal
    q = C @ (D @ P)
    return q, P


#%% setting up inital matrices and vecotrs 

P_BC = np.array([pleft, pleft, pleft, pleft, pright, pright, pright, pright])
C = conductance(edge_widths)
Abb, Abn, Anb, Ann = A_matrices( C, Db, Dn)
times = [t]
q,P = solve_qP()

q0 = q.copy()# solving origional q and widths to plot later
width0 = edge_widths

qn = D.T @ C @ D @ P
s_plus = np.zeros(num_nodes)  # Chemical plus at nodes
s_minus = np.zeros(num_nodes)  # Chemical minus at nodes
S_plus = np.zeros(len(edges))  # Chemical plus in edges
S_minus = np.zeros(len(edges))  # Chemical minus in edges

#%% chemical functions 

def s_initial():
    """Initialize chemicals at output nodes based on pressure error."""
    for i, a in  enumerate(output_nodes):
       
        error = P[a] - des_P[i]
        if error > 0:
            s_plus[a] = lam * (P[a] - des_P[i])
        elif error < 0:
            s_minus[a] = lam * (des_P[i] - P[a])
            
def update_s():
    """ distributing the chemical trough the network by advection """
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
delta = 0.1
def learning(y, outP):
    """function that sets the desried output pressure for every entry 
    to nutch the output pressure towards the right output node"""
    # start from current output pressure
    desP = np.ones(3)* np.mean(outP)
    desP -= delta  # slightly lower output pressure
    desP[y] += 3 *delta # slightly nude right output node 
    return desP

S_plus_list = np.zeros((max_steps, num_edges))
S_minus_list = np.zeros((max_steps, num_edges))
q_list = np.zeros((max_steps, num_edges))
widht_list = np.zeros((max_steps, num_edges))
Plist =np.zeros((max_steps, len(output_nodes)))

# keeping track of interval between releases
releases = 0
release_time = []
previous_release_step = 0

#%% cell of dath that cotains the big for loop
print(f'starting, Nx,Ny = {Nx}, {Ny},  max_steps = {max_steps}')
    
for i in range(len(X_training)):
        
        #select output pressure and right output node
        P_BC = np.concatenate((np.zeros(4), X_training[i]))
        y = y_training[i]
    
        # find desired output pressure 
        q, P = solve_qP() 
        outP = P[output_nodes]
        des_P = learning(y_training[i], P_BC)
        
        
        # reset chemical 
        s_plus.fill(0)
        s_minus.fill(0)
        
        print(f"Training entry {i}/75:")
        print(f"  iris label: {y_training[i]}")
        print(f"  Output pressures: {outP}")
        print(f"  Desired pressures: {des_P}")

        for j in range(training_steps):
            #check if the chemical is drained
            total_chemical = np.sum(s_plus) + np.sum(s_minus) 
         
            if total_chemical == 0: # flip boundary pressure
                P_BC *= -1
                if np.any(P_BC[5:] * des_P < 0):
                    des_P *= -1
               
                C = conductance(edge_widths)
                D, Db, Dn = D_matrix(edges, num_nodes, num_edges, input_nodes, internal_nodes)
                Anb = Dn.T @ C @ Db
                Ann = Dn.T @ C @ Dn   
                q, P = solve_qP()
                
                # relase chemical 
                s_initial()
                releases += 1
                release_time.append(steps - previous_release_step)
                previous_release_step = steps
                
            # Update chemical flow
            S_plus, S_minus = update_s()  
            
            # Update edge widths based on chemical flow
            dC_dt = xi * (S_plus - S_minus)
            dr_dt = 2 * edge_lengths / pi * 1 / (edge_widths**3) * dC_dt
            dt = 1
            dr = dr_dt * dt
            edge_widths = np.maximum(edge_widths + dr, 0.1)
            
            # Update matrices and solve
            D, Db, Dn = D_matrix(edges, num_nodes, num_edges, input_nodes, internal_nodes)
            C = conductance(edge_widths)
            Anb = Dn.T @ C @ Db
            Ann = Dn.T @ C @ Dn   
            q, P = solve_qP()
            
            # Store useful data
            S_plus_list[steps] = S_plus
            S_minus_list[steps] = S_minus
            q_list[steps] = q
            widht_list[steps] = edge_widths 
            Plist[steps] = P[output_nodes]
            
            steps +=1

S_total_list = S_plus_list - S_minus_list
#%% test model 

C   = conductance(edge_widths)
D,Db,Dn = D_matrix(edges, num_nodes, num_edges,
                       input_nodes, internal_nodes)
Anb = Dn.T @ C @ Db
Ann = Dn.T @ C @ Dn

def predict(x_sample):
    """for a x (testing ) sample give the output pressure"""
    P_BC = np.zeros(num_nodes)
    P_BC[right_nodes] = x_sample
    P = np.zeros(num_nodes)
    P[input_nodes] = P_BC[input_nodes]
    P_internal = np.linalg.pinv(Ann) @ (-Anb @ P_BC[input_nodes])
    P[internal_nodes] = P_internal
    
    return P[output_nodes]


# test testing data
y_pred = np.array([ np.argmax(predict(x_s)) for x_s in X_testing ]) 
print(f'testing data: {y_testing}')
print(f'testing predited: {y_pred}')
acc    = np.mean(y_pred == y_testing)
print(f"Test accuracy : {acc*100:.2f}%")



# test traingn data
y_pred = np.array([ np.argmax(predict(x_s)) for x_s in X_training ])
print(f'training data: {y_training}')
print(f' training predited: {y_pred}')
acc    = np.mean(y_pred == y_training)
print(f"Training accuracy : {acc*100:.2f}%")


#%% making histogram of 

data = release_time[1:]
print(
    f"releases: total {len(data)}, mean = {np.mean(data):.1f}, "
    f"max = {np.max(data)}, min = {np.min(data)}, "
    f"std = {np.std(data, ddof=1):.1f}")


# Number of integer‑sized bins spanning from min to max
nbins = int(np.max(data) - np.min(data))

# Compute histogram
counts, bin_edges = np.histogram(
    data ,
    bins=nbins,
    range=(np.min(data), np.max(data)))

# Calculate centers and width
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(
    bin_centers,
    counts,
    width=bin_width,        
    edgecolor='k',
    alpha=1)

# Add mean line
avg_interval = np.mean(data)
plt.axvline(
    avg_interval,
    color='r',
    linestyle='--',
    label=f'Average: {avg_interval:.1f} steps')


ax = plt.gca()
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Labels, legend, grid, limits
plt.xlabel('Time between releases (steps)')
plt.ylabel('Count')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(np.min(data), np.max(data))
plt.show()
plt.savefig('sinus_chemical_release', dpi=300, bbox_inches='tight')


#%% plotting pressure 
P1 = np.abs(Plist[::, 0])
P2 = np.abs( Plist[::,1])
P3 = np.abs( Plist[::,2])

plt.figure()
plt.title('pressure')
plt.plot(np.arange(0, len(P1),1), P1, color = 'k')
plt.plot(np.arange(0, len(P2),1), P2, color = 'b')
plt.plot(np.arange(0, len(P3),1), P3, color = 'r')

# plt.xlim((0, steps))
plt.xlabel('steps')
plt.ylabel('Pressure')
plt.show()

#%% plotting final flux and density function
#plotting flow network 
plot(q0, '', nodes, edges, width0, filename= 'start_flow_iris')
plot(q, f'Final Network, steps ={steps}', nodes, edges, edge_widths)
print(f'total time is:{time.time()-start:.2f}, releases = {releases}')

#%% animate chemical 

skip = 1  # controling how many simulated frames are animated
N_frames = int(len(P1)) - 1


aspect = Nx / Ny  # Data aspect ratio
fig_height = 8    # Arbitrary height in inches
fig_width = fig_height * aspect  # Width adjusted to match data aspect
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

widths = widht_list[0]
width_norm = widths/max(widths)
widths= 2 * width_norm +0.3

lc = mc.LineCollection(edge_lines, cmap='cool', alpha = 0.8, linewidths = widths)
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
    S =  np.abs(S_total_list[t])
    S = S> 0

    # Compute linewidths
    widths = widht_list[t]  # Note: 'widht_list' seems to be a typo; should be 'width_list'
    width_norm = widths / max(widths)
    widths = 2 * width_norm + 0.3
    
    
    # Clear previous LineCollection
    for coll in ax.collections:
        if isinstance(coll, mc.LineCollection):
            coll.remove()
    
    # Create a new LineCollection with the pre-defined colormap
    lc = mc.LineCollection(edge_lines, cmap='cool', linewidths=widths)
    lc.set_array(S)
    ax.add_collection(lc)
    
    # Update time text
    time_text.set_text(f'Steps: {t}')
    ax.set_ylim(0, Ny)  # Use ax.set_ylim instead of plt.ylim for consistency
    
    return lc, time_text

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=N_frames, interval=200)

# # Save the animation
# anim.save('chemical.gif', writer=animation.PillowWriter(fps=5))
