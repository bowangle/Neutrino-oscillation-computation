"""This code is tasked to compute on a quantum emulator (in opposition to classical computation) the two flavor approximation of the oscillation of neutrino with the two body interraction.
This code regroupe every function and methode related to the quantum emulation of the quantum circuit related to the Hamiltonian introduced by Amitrano and al in the following publication: (https://journals.aps.org/prd/abstract/10.1103/PhysRevD.107.023007).
The code emulate the propagation of multiple beam of 1 neutrino through time with the possibility to have interraction between neutrino.
The default coefficient are able to reproduce the result from the publication.
At the end of this file, you will an "Application" section where I explain every parameter you can modify easily and how to run the code.

code dated of:
11/07/2024

Here are the vesion of python as well as the package used:

Python          3.9.5
scipy           1.13.0
numpy           1.26.4
matplotlib      3.4.2

qiskit          1.0.2
qiskit_aer      0.14.1
"""

##Include
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import Statevector, Operator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from qiskit_aer import AerSimulator
from qiskit_aer import StatevectorSimulator
import time

import matplotlib

from qiskit.primitives import Sampler

from qiskit.quantum_info import partial_trace

from qiskit.circuit.library import XXPlusYYGate
from qiskit.circuit.library import RXXGate
from qiskit.circuit.library import RYYGate
from qiskit.circuit.library import RZZGate
from qiskit.circuit.library import RXGate
from qiskit.circuit.library import RYGate
from qiskit.circuit.library import RZGate
from qiskit.circuit.library import QFT
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library import PhaseGate
from scipy.linalg import expm,logm

##U1 propagator
"""This section present the fonction that construct the circuit corresponding to 1 step of the one body term of the Hamiltonian.

The coefficient are directly input into the gate.

The arguments needed to build the U1 propagator are:
    * dt (float): Time step of the propagator
    * theta(float): Mixing angle for a specific 2 flavor case
    * N (int): Number of neutrino == Number of qubit

These functions will be used by the "U_propagator_1_step" function. If you want to choose one specifically you have to change it by hand inside of it.
"""

def U1_propagator_trotter_1_step(dt,theta,N):#constuct the propagator for U1 for 1 dt
    """Function that create an approximate version of the U1 propagator. It return the circuit corresponding to a single time step of size dt. It apply the gate to the N neutrino present in the ciruit.

    Args:
        dt (float): Time step of the propagator
        theta(float): Mixing angle for a specific 2 flavor case
        N (int): Number of neutrino == Number of qubit

    Returns:
        U1_troter (QuantumCircuit): Circuit corresponding to the propagation of the U1 propagator over a time dt
    """
    U1_troter=QuantumCircuit(N,N)
    for i in range(N):
        U1_troter.rz(-dt/N*np.sin(2*theta),i)
        U1_troter.rx(dt*np.cos(2*theta),i)
    U1_troter.barrier()
    return U1_troter

def U1_propagator_exact_1_step(dt,theta,N):#constuct the propagator for U1 for 1 dt, methode 2
    """Function that create an exact version of the U1 propagator. It return the circuit corresponding to a single time step of size dt. It apply the gate to the N neutrino present in the ciruit.

    Args:
        dt (float): Time step of the propagator
        theta(float): Mixing angle for a specific 2 flavor case
        N (int): Number of neutrino == Number of qubit

    Returns:
        U1_troter (QuantumCircuit): Circuit corresponding to the propagation of the U1 propagator over a time dt
    """
    C=1/N
    delta=np.pi/4-np.arctan(np.tan(C*dt)*np.cos(2*theta))
    beta=-np.pi/4-np.arctan(np.tan(C*dt)*np.cos(2*theta))
    gamma=2*np.arccos((np.cos(C*dt))/(np.cos(-beta/2-delta/2)))
    U1_troter=QuantumCircuit(N,N)
    for i in range(N):
        U1_troter.rz(beta,i)
        U1_troter.ry(gamma,i)
        U1_troter.rz(delta,i)
    U1_troter.barrier()
    return U1_troter

##U2 propagator
"""This section present the fonction that construct the circuit corresponding to 1 step of the two body interraction term of the Hamiltonian.

The coefficient are directly input into the gate.

The arguments needed to build the U2 propagator are:
    * dt (float): Time step of the propagator
    * N (int): Number of neutrino == Number of qubit

These functions will be used by the "U_propagator_1_step" function. If you want to choose one specifically you have to change it by hand inside of it.
"""

def Jij(N): #return the matrice of Jij
    """This function compute the coefficient of the two body interraction term. It pre calculate all the coefficient and store them inside an array for easier acces.

    Args:
        N (int): Number of neutrino == Number of qubit

    Returns:
        J (array of size N*N): Array containing the coefficient corresponding to the interraction between two neutrino i and j. J[i,j] give this interracation.
    """

    J=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            J[i,j]=(1/N)*(1-np.cos(np.arccos(0.9)*np.abs(i-j)/(N-1)))
    return J

def U2_propagator_1_step(dt,N):#constuct the propagator for U2 for 1 dt, methode 1
    """Function that create a version of the U2 propagator using as few gate as possible (excluding the XX+YY gate that is generaly not available). It return the circuit corresponding to a single time step of size dt. It apply the gate to the whole system of qubit doing triangulare superior permutation.

    Args:
        dt (float): Time step of the propagator
        N (int): Number of neutrino == Number of qubit

    Retunrs:
        U2 (QuantumCircuit): Circuit corresponding to the propagation of the U2 propagator over a time dt

    """
    jij=Jij(N)
    U2=QuantumCircuit(N,N)
    for i in range(N):
        for j in range(i+1,N):#i<j
            U2.rzz(2*dt*jij[i,j],i,j)
            U2.ryy(2*dt*jij[i,j],i,j)
            U2.rxx(2*dt*jij[i,j],i,j)
            U2.barrier()
    return U2

def U2_propagator_1_step_v2(dt,N):#smaller trotter approximation
    """Function that create a version of the U2 propagator using as few gate as possible (based on the gate disponible in qiskit). It return the circuit corresponding to a single time step of size dt. It apply the gate to the whole system of qubit doing triangulare superior permutation.

    Args:
        dt (float): Time step of the propagator
        N (int): Number of neutrino == Number of qubit

    Retunrs:
        U2 (QuantumCircuit): Circuit corresponding to the propagation of the U2 propagator over a time dt

    """
    #constuct the propagator for U2 for 1 dt, methode 2
    jij=Jij(N)
    U2=QuantumCircuit(N,N)
    for i in range(N):
        for j in range(i+1,N):#i<j
            coef=2*dt*jij[i,j]

            U2.rzz(coef,i,j)
            U2.append(XXPlusYYGate(2*coef),[i,j])
            U2.barrier()
    return U2

def U2_propagator_1_step_optimal(dt,N):#constuct the propagator for U2 for 1 dt, methode 3
    """Function that create a version of the U2 propagator using as few usual gate as possible (we assume we don't have direct access to RZZ, RXX and RYY) and we minimized the number of two qubit gate. It return the circuit corresponding to a single time step of size dt. It apply the gate to the whole system of qubit doing triangulare superior permutation.

    Args:
        dt (float): Time step of the propagator
        N (int): Number of neutrino == Number of qubit

    Retunrs:
        U2 (QuantumCircuit): Circuit corresponding to the propagation of the U2 propagator over a time dt

    """
    jij=Jij(N)
    U2=QuantumCircuit(N,N)
    for i in range(N):
        for j in range(i+1,N):#i<j
            phi=-2*dt*jij[i,j]-np.pi/2

            U2.p(np.pi/2,i)
            U2.rz(-np.pi/2,i)
            U2.rz(-np.pi/2,j)
            U2.cx(j,i)
            U2.rz(-phi,i)
            U2.ry(phi,j)
            U2.cx(i,j)
            U2.ry(-phi,j)
            U2.cx(j,i)
            U2.rz(np.pi/2,i)
            U2.barrier()
    return U2

##Others circuit parts
"""These function are to the circuit parts of the circuit. The are related to the initialisation and the measurement of the circuit.
"""

def init_propagator(phi0):#initialise the circuit to it's default state
    """This function compute the circuit related to the initial state of the circuit.

    Args:
        phi0 (list of int): List of 0 and 1 describing the initial state of the circuit

    Returns:
        Return the circuit correponding to this initial state
    """
    N=len(phi0)
    U_init=QuantumCircuit(N,N)
    for i in range(N):
        if phi0[i]==1:
            U_init.x(i)
    U_init.barrier()
    return U_init



def add_measurment(U,phi0,measurement="Z"):#add the measurement, and change the measurement basis
    """This function add the circuit corresponding to the measurement part of the circuit to the circuit U. Before applying the measurement, we do the appropriate measurment basis changes. It create a copy of the circuit. It's not modifying the circuit we send to it. It's not in place calculation.

    Args:
        U (QuantumCircuit): Circuit we want to add the measurement operation
        phi0 (list of int): Just for the number of neutrino (we could have send N instead)
        measurement (str): Either X,Y or Z, indicate which basis we are measuring in

    Returns:
        U_m (QuantumCircuit): output circuit to which we just add the measurement circuit in the correct basis

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    This is not correct for X_i measurement as we are measuring X_2=XXXXX instead of X_2=IXIII
    same for Y_i
    It's actually surprising we get the same result for cdzz
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    N=len(phi0)
    U_m=U.copy()#we need this!!!
    if measurement=="X":
        for i in range(N):
            U_m.h(i)
    if measurement=="Y":
        for i in range(N):
            U_m.p(-np.pi/2,i)
            U_m.h(i)
    for j in range(N):
        U_m.measure(j,j)
    return U_m

##whole propagator
"""These function are related to the assembling of the different parts of the circuit.
"""

def U_propagator_1_step(dt,theta,N):  #concatenate both U1 and U2 into a circuit
    """This function construct the whole propagator for a given step of size dt by puting both the U1 and U2 circuit right after another.

    Args:
        dt (float): Time step of the propagator
        theta(float): Mixing angle for a specific 2 flavor case
        N (int): Number of neutrino == Number of qubit

    Returns:
        U (QuantumCircuit): Circuit corresponding to the propagation of the U2*U1 propagator over a time step dt, for N neutrino.
    """
    U1=U1_propagator_exact_1_step(dt,theta,N)

    U2=U2_propagator_1_step(dt,N)
    #U2=U2_propagator_1_step_optimal(dt,N)
    U=U1.compose(U2)
    return U

def U_propagator_rec(phi0,U_1_step,U_last_step,IS_BEGINING=False):
    """This function add 1 step of propagator of dt. If we are at the first step, it initialise the circuit to the initial condition.

    Args:
        phi0 (list of int): List of 0 and 1 describing the initial state of the circuit
        U_1_step (QuantumCircuit): Circuit doing the propagation for one step of dt
        U_last_step (QuantumCircuit): Circuit we had for the last step, not used if we are at the first step
        IS_BEGINING (boolean): Use to indicate if we already had done any step before. True if this is the first call, False otherwise.

    Returns:
        U (QuantumCircuit): Circuit combining the last step with one more step or the initial circuit depending on IS_BEGINING
    """
    if IS_BEGINING:
        U=init_propagator(phi0)
    else:
        U=U_last_step.compose(U_1_step)
    return U


##simulation:
def simulate_circuit_all_t(k_step,dt,theta,N,phi0,nb_shots,measurement="Z"):
    """This function do the sampling of the circuit for each time step. Then it format the result into an array and return it.

    Args:
        k_step (int): Number of time step we have to measure
        dt (float): value of a time step
        theta(float): Mixing angle for a specific 2 flavor case
        N (int): Number of neutrino/qubit in the circuit
        phi0 (list of int): List of 0 and 1 describing the initial state of the circuit
        nb_shots (int): Parameter for the sampler of the quantum emulator to choose the number of sampling to do
        measurement (str): Either X,Y or Z, indicate which basis we are measuring in

    Returns:
        tab_result (array of shape (2**len(phi0),k_step)): Array storing the occurence of each state for all the diffent time. tab_result(i,t) will give the occurence of the state id i at the time t
        l_state (list of (str of len N)): List of the the state corresponding to each index of the tab_result array
    """
    #create the circuit and simulate it
    #return the result and the order of the state ["0000","0001",...]
    def generate_binary_ordered_str_permutation(n):#create a dictionnary that given a key of the sort: "1001" return the id for the dimension 1 of the resulte table, it also return the list of all the state ordered
        """This function create a dictionnary that given a state result the id of the state on the tab_result array. It also return the list of theses states.

        Args:
            n (integer): number of qubits

        Returns:
            dicl (dic): The dictionnary that given a state as a key return the corresponding id for this state
            l (list of str): List of all the state ordered by id
        """
        def next_step(last_step):
            """Function that compute the permutation list of element in 0,1 of n+1 element given the one at n element."""
            l_out=[]
            for i in last_step:
                l_out.append('0'+i)
                l_out.append('1'+i)
            return l_out
        l=['']
        for i in range(n):
            l=next_step(l)
        l=np.sort(l)
        dicl={l[i]: i for i in range(len(l))}
        return dicl,l

    def l_string_reverse(l):
        """This function reverse the element order in a list."""
        for i in range(len(l)):
            l[i]=l[i][::-1]
        return l

    tab_result=np.zeros((2**len(phi0),k_step))#table of result, table[id_state,id_step]=nb_event
    key_to_itab_index,l_state=generate_binary_ordered_str_permutation(len(phi0))

    #we need to correct the order of state done by qiskit in l_ordered_state, we just need to reverse the string
    l_state=l_string_reverse(l_state)

    U_1_step=U_propagator_1_step(dt,theta,N)
    U_n=U_propagator_rec(phi0,U_1_step,0,IS_BEGINING=True)

    U_nm=add_measurment(U_n,phi0,measurement)
    tps1=time.perf_counter()

    for i in range(k_step):
        result = AerSimulator().run(U_nm,shots=nb_shots).result()
        U_n=U_propagator_rec(phi0,U_1_step,U_n,IS_BEGINING=False)
        U_nm=add_measurment(U_n,phi0,measurement)
        statistics = result.get_counts()
        L_key=list(statistics.keys())

        for j in range(len(L_key)):
            tab_result[key_to_itab_index[L_key[j]],i]=statistics[L_key[j]]
        if i%50==0 and i!=0:#just to print
            tps2=time.perf_counter()
            print("step ",i,"in: ",tps2-tps1,"s")
            tps1=time.perf_counter()
    return tab_result,l_state

##operator evaluation

def evaluate_operator_Z(tab_result,state_order):
    """The function evaluate a single qubit operator (either Z,X or Y). It measure this operator for every neutrino and every time step. It return an array storing this result. This function should only work for Z measurement.

    Args:
        tab_result (array of shape (2**len(phi0),k_step)): Array storing the occurence of each state for all the diffent time. tab_result(i,t) will give the occurence of the state id i at the time t
        l_state (list of (str of len N)): List of the the state corresponding to each index of the tab_result array

    Retunrs:
        tab_out (array of shape(N,nb_t): Array storing the value of the operator Z for all neutrino and for each time. tab_out[i,t] give the value Z_i(t) the value of the opera
    """
    #Only work for Z operator measurment, given the result table and the state order it computer the the table Zi(t)=Z[i,t] with i being the index of the neutrino
    n_state,ndt=np.shape(tab_result)
    nb_qbit=len(state_order[0])
    nb_sample=sum(tab_result[:,0])

    tab_out=np.zeros((nb_qbit,ndt)) #tab[id_qbit,idt]

    for t in range(ndt):
        for j in range(nb_qbit):
            s=0
            for i in range(n_state):
                if state_order[i][j]=="1":
                    s=s-1*tab_result[i,t]
                else:
                    s=s+1*tab_result[i,t]
            tab_out[j,t]=s/nb_sample
    return tab_out

def evaluate_operator_Zi_Zj(tab_result,state_order):
    """The function evaluate a two qubit operator ZiZj. It measure this operator for every couple of two neutrino and every time step. It return an array storing this result. Only work for ZZ measurement.

    Args:
        tab_result (array of shape (2**len(phi0),k_step)): Array storing the occurence of each state for all the diffent time. tab_result(i,t) will give the occurence of the state id i at the time t
        l_state (list of (str of len N)): List of the the state corresponding to each index of the tab_result array

    Retunrs:
        tab_out (array of shape (nb_qubit,nb_qubit,nbt): array storing the Zij(t) result. tab_out[i,j,idt] give the value of the Zij operator at the time step idt.
    """
    n_state,ndt=np.shape(tab_result)
    nb_qbit=len(state_order[0])
    nb_sample=sum(tab_result[:,0])

    tab_out=np.zeros((nb_qbit,nb_qbit,ndt)) #tab[id_qbit,idt]

    ZZ_operator={"00":1,"01":-1,"10":-1,"11":1}
    for t in range(ndt):
        for j in range(nb_qbit):
            for k in range(nb_qbit):
                s=0
                for i in range(n_state):
                    current_state=state_order[i][j]+state_order[i][k]
                    s=s+ZZ_operator[current_state]*tab_result[i,t]
                tab_out[j,k,t]=s/nb_sample
    return tab_out #tab_Zij_out[i,j,:]=Zi_Zj


##QPE quantum phase estimation
"""This section focus on another problem. We wanted to experiment a bit with the quantum phase experiment (QPE). In this section the goal is to use our propagator explained earlier and extract the eigenvalue of the Hamiltonian. It's not exactly the QPE function, we had to do some modification to it to actually be able to predict the eigenvalue of the Hamiltonian.
"""

#this section contain fonction related to the QPE methode

#first define the controled propagator:
def U2_propagator_1_step_QPE(dt,theta,N,m,id_controled,coeff=1):
    """This function compute the U2 propagator for the QPE for one step. We had to use a different function than before because we have to controle this operator.

    Args:
        dt (float): value of a time step
        theta(float): Mixing angle for a specific 2 flavor case
        N (int): Number of neutrino/qubit in the circuit
        m (int): Number of ancillary qubit, qubit that hold the eingenvalue prediction in our case.
        id_controled (int): id of the qubit we want to  control it to
        coeff (float): Coefficient of the two body interraction

    Returns:
        U2 (QuantumCircuit): The circuit corresponding to one step of controle U2.
    """

    jij=Jij(N)
    id_controle=id_controled
    U2=QuantumCircuit(N+m,m)
    for i in range(N):
        for j in range(i+1,N):#i<j
            #phase opposite
            c1=RZZGate(-2*coeff*dt*jij[i,j]).control(1)
            c2=RYYGate(-2*coeff*dt*jij[i,j]).control(1)
            c3=RXXGate(-2*coeff*dt*jij[i,j]).control(1)

            U2.append(c1,[id_controled,m+i,m+j])
            U2.append(c2,[id_controled,m+i,m+j])
            U2.append(c3,[id_controled,m+i,m+j])
    return U2

def U1_propagator_exact_1_step_QPE(dt,theta,N,m,id_controled,coeff=1):
    """This function compute the U1 propagator for the QPE for one step. We had to use a different function than before because we have to controle this operator.

    Args:
        dt (float): value of a time step
        theta(float): Mixing angle for a specific 2 flavor case
        N (int): Number of neutrino/qubit in the circuit
        m (int): Number of ancillary qubit, qubit that hold the eingenvalue prediction in our case.
        id_controled (int): id of the qubit we want to  control it to
        coeff (float): Coefficient of the one body term

    Returns:
        U2 (QuantumCircuit): The circuit corresponding to one step of controle U2.
    """
    print(dt,theta,N,m,id_controled)
    id_controled=id_controled
    C=1/N
    delta=np.pi/4-np.arctan(np.tan(-C*coeff*dt)*np.cos(2*theta))
    beta=-np.pi/4-np.arctan(np.tan(-C*coeff*dt)*np.cos(2*theta))
    gamma=2*np.arccos((np.cos(-C*coeff*dt))/(np.cos(-beta/2-delta/2)))
    U1_troter=QuantumCircuit(N+m,m)
    for i in range(N):
        c1=RZGate(beta).control(1)
        c2=RYGate(gamma).control(1)
        c3=RZGate(delta).control(1)
        U1_troter.append(c1,[id_controled,i+m])
        U1_troter.append(c2,[id_controled,i+m])
        U1_troter.append(c3,[id_controled,i+m])
    return U1_troter

def init_propagator_QPE(phi0,m):
    """This function initialise the initiale state of the QPE process.
        Args:
            phi0 (list of int): List of 0 and 1 describing the initial state of the circuit
            m (int): Number of ancillary qubit, qubit that hold the eingenvalue prediction in our case.

        Returns:
            U_init (QuantumCircuit): The circuit corresponding to the initial state
    """
    N=len(phi0)
    U_init=QuantumCircuit(N+m,m)
    for i in range(N):
        if phi0[i]==1:
            U_init.x(m+i)
    return U_init

def add_measurment_QPE(circuit,m):
    """This funciton add the measurement to the circuit. Operation in place (but we still use the

        Args:
            circuit (QuantumCircuit): Circuit we want to measure
            m (int): Number of ancillary qubit, qubit that hold the eingenvalue prediction in our case.

        Returns:
            circuit (QuantumCircuit): Circuit to which we add the measurement
    """
    for i in range(m):
        circuit.measure(i,i)
        #circuit.measure(i,m-i-1)
    return circuit

def add_h_layer_QPE(circuit,m):
    """This circuit add the H layer on all ancillary qubit. it's part of the QPE methode.

        Args:
            circuit (QuantumCircuit): Circuit we want to add the H layer
            m (int): Number of ancillary qubit, qubit that hold the eingenvalue prediction in our case.

        Returns:
            circuit (QuantumCircuit): Circuit to which we add the measurement
    """
    #add the layer of H
    for i in range(m):
        circuit.h(i)
    return circuit

def add_phase(U,coeff,N,m,id_controled):
    """Fonction that add a phase to the QPE qubit, it's necessary to be able to get back the eigenvalue of H.

    Args:
        U (QuantumCircuit): Circuit to which we will add the phase
        coeff (float): value of the phase (factor 2)
        m (int): number of ancillary qubits
        id_controled (int): qubit we need the phase to be controlled by

    Returns:
        U (quantumCircuit): Circuit to which we just add the controlled phase gate

    """
    #add a phase to the circuit
    U.barrier()

    #only add phase to a single ligne, it would be wrong to add it to every single line
    g1=PhaseGate(coeff).control(1)
    g2=RZGate(-coeff).control(1)

    U.append(g1,[id_controled,m])
    U.append(g2,[id_controled,m])
    U.barrier()
    return U

def assemble_circuit_for_QPE(phi0,theta,m,dt,t_max,E_min,E_max): #construct the circuit for the QPE, return it
    """Fonction that assemble all the part of the QPE circuit together.
    E_min and E_max are choosen by hand for now, but their is way to compute E_max quickly

    Args:
        phi0 (list of int): List of 0 and 1 describing the initial state of the circuit
        theta(float): Mixing angle for a specific 2 flavor case
        m (int): Number of ancillary qubit, qubit that hold the eingenvalue prediction in our case.
        dt (float): value of a time step
        t_max (float): scale factor in this case, we are relatively free on the value, you should not change it
        E_min (float): minimal value of the spectrum of H
        E_max (float): maximal value of the spectrum of H

    Returns:
        U (quantumCircuit): The whole circuit for the QPE methode.
    """

    N=len(phi0)
    l_t=[i*dt for i in range(1,int(1+np.ceil(t_max/dt)))]
    #print(l_t)
    n_step=len(l_t)
    print("we need to do",n_step)

    L_circuit=[]#end with a len of m, this will contain the list of 1 dt step controled U controling the qbits of index i

    alpha=2*np.pi/(E_max-E_min)
    beta=-2* 2*np.pi*(E_min/(E_max-E_min))

    #compute L_circuit
    for i in range(m):
        a=U1_propagator_exact_1_step_QPE(dt,theta,N,m,i,coeff=alpha)
        b=U2_propagator_1_step_QPE(dt,theta,N ,m,i,coeff=alpha)
        U_1_step= a.compose(b)

        U_1_step=add_phase(U_1_step,beta*dt,N,m,i)

        U_controled_t=QuantumCircuit(N+m,m)
        for j in range(n_step):
            U_controled_t=U_controled_t.compose(U_1_step)

        U_controled_ti=QuantumCircuit(N+m,m)
        U_controled_ti=U_controled_ti.compose(U_controled_t)

        #test
        #U_controled_ti=add_phase(U_controled_ti,beta*dt,N,m,i)

        for k in range(1,2**(i)):
            U_controled_ti=U_controled_ti.compose(U_controled_t)
        L_circuit.append(U_controled_ti)


    U=QuantumCircuit(N+m,m)
    U_temp=init_propagator_QPE(phi0,m)
    U=U.compose(U_temp)
    U=add_h_layer_QPE(U,m)
    #put all the U^(2i)
    for i in range(len(L_circuit)):
        U=U.compose(L_circuit[i])

    #add the QFT gate
    QFTGate=QFT(m,inverse=True)
    U.append(QFTGate,[i for i in range(m)])
    U.barrier()

    U=add_measurment_QPE(U,m)
    print("circuit is ready for QPE")
    return U

def simulate_QPE(U,nb_shot,m):
    """This function do the simulation on the quantum emulator of the circuit and do the sampler of the result.

    Args:
        U (QuantumCircuit): Circuit we have to sample
        nb_shot (int): Sampling parameter that indicate how many time we run the circuit. A too low value may lead significative statistical fluctuation.
        m (int): Number of ancillary qubit, qubit that hold the eingenvalue prediction in our case.

    Returns:
        l_key (list): List of state id in base 10. ex: [0,2,4,5,...63]
        l_proba (list): List of the proba associated to each of l_key state (sum should add to one)
    """
    #simulate the circuit for the QPE process, return the list of state and proba associated to these state
    print("start sampling:")
    #result = AerSimulator().run(U,shots=nb_shot).result() not working with crz, cry and crx
    result=Sampler().run(U,shots=nb_shot).result()
    #print(result)
    dict_result=result.quasi_dists[0]
    #print(dict_result)

    result_keys=list(dict_result.keys())
    result_items=list(dict_result.items())
    #print("ici", result_items)
    #print(result_items)
    l_key=[]
    l_proba=[]
    for i in range(len(result_keys)):
        a,b=result_items[i]
        l_key.append(a)
        l_proba.append(b)
    return l_key,l_proba

def treatment_result_QPE(l_key,l_proba,m,E_min,E_max,t_max):
    """As we had to rescale the Hamiltonian to make eingenvalue enter the range 0,1; we have to scale back to the initial range of value. This function compute for each l_key the corresponding engenvalue back scalling back the value. Then it filter the most probable state and print them. It also print all the result to let us check the value. At the end it return the whole result.

    Args:
        l_key (list): List of state id in base 10. ex: [0,2,4,5,...63]
        l_proba (list): List of the proba associated to each of l_key state (sum should add to one)
        m (int): Number of ancillary qubit, qubit that hold the eingenvalue prediction in our case.
        E_min (float): minimal value of the spectrum of H
        E_max (float): maximal value of the spectrum of H
        t_max (float): scale factor in this case, we are relatively free on the value, you should not change it

    Returns:
        l_proba (list of float): List of the proba associated to each of l_key state (sum should add to one). We didn't modify it in the process
        l_lambda (list of float): list that contain the eigenvalue of H. It has the same order has l_key and l_proba.
        l_label (list of str): Give the form of the actual output of the state in binary. (list of ["10100011","01110110"] for example

        This function also print a result list using filter_result
    """

    def filter_result(l_proba,l_lambda,l_state):
        """This function is used to quickly sort out all the most probable state using a threshold. In this case it discard all the state with less than 0.01=1% of happening. At the end it organise the result in a list that we print.

        Args:
            l_proba (list): List of the proba associated to each of l_key state (sum should add to one)
            l_lambda (list of float): list that contain the eigenvalue of H. It has the same order has l_key and l_proba.
            l_label (list of str): Give the form of the actual output of the state in binary. (list of ["10100011","01110110"] for example

        Returns:
            void
            print:
                l_out (list of list): list of [l_proba[i],l_lambda[i],l_state[i]] for all i where l_proba[i]>threshold
        """
        print(len(l_proba))
        print(len(l_lambda))
        print("the result are:")
        l_out=[]
        for i in range(len(l_proba)):
            if l_proba[i]>=0.01:
                l_out.append([l_proba[i],l_lambda[i],l_state[i]])
        print(l_out)


    def state_measured_to_phi(l_key,m):
        """Using the key this function compute the actual value of the angle in 0,1 intervale, before scaling bak to E_min E_max. It return this list as well as the list of the label of each state in binary.

        Args:
            l_key (list): List of state id in base 10. ex: [0,2,4,5,...63]
            m (int): Number of ancillary qubit, qubit that hold the eingenvalue prediction in our case.

        Return:
            l_phi (list of float): phase in 0,1 before rescale in E_min, E_max
            l_label (list of str): Give the form of the actual output of the state in binary. (list of ["10100011","01110110"] for example
        """
        l_phi=[]
        l_label=[]
        diviser=2**m
        for i in l_key:
            temp=bin(i)[2:]              #convert to a string of binary
            if len(temp)!=m:               #uniformise the 0 of the string
                temp="0"*(m-len(temp))+temp
            l_label.append(temp)

            l_phi.append(i/diviser)
        return l_phi,l_label

    l_phi_in_01,l_label=state_measured_to_phi(l_key,m)
    print(l_phi_in_01)
    # We can set the number of bins with the *bins* keyword argument.
    print("output:")
    print("key: ",l_key)
    print("theta: ", l_phi_in_01)

    print(E_max)
    print(E_min)
    l_lambda=[i*(E_max-E_min)/t_max+E_min for i in l_phi_in_01]

    print("prediction eigenvalue: ",l_lambda)

    print("proba: ",l_proba)
    print("label: ",l_label)

    id_max=l_proba.index(max(l_proba))
    print("predicted value: ",l_key[id_max],l_phi_in_01[id_max],l_lambda[id_max],l_proba[id_max])

    l_lambda=[str(round(i,3)) for i in l_lambda]

    filter_result(l_proba,l_lambda,l_label)

    return l_proba,l_lambda,l_label

def main_QPE(phi0,theta,m,dt,nb_shot,E_max,E_min):
    """This function is the main function of QPE. This is where I handwrite some parameter and the one function we call to do the computation of the QPE methode. It return the result at the end. It first construct the circuit, then simulate it and finaly analyse the results.

    Args:
        phi0 (list of int): List of 0 and 1 describing the initial state of the circuit
        theta(float): Mixing angle for a specific 2 flavor case
        m (int): Number of ancillary qubit, qubit that hold the eingenvalue prediction in our case.
        dt (float): value of a time step
        nb_shot (int): Sampling parameter that indicate how many time we run the circuit. A too low value may lead significative statistical fluctuation.
        E_min (float): minimal value of the spectrum of H. We don't need the exact value but a rough estimate of it's value, anything smaller than E_min will work but too small value will decrease the resolution.
        E_max (float): maximal value of the spectrum of H. We don't need the exact value but a rough estimate of it's value, anything bigger thatn E_max will work but too big value will decrease the resolution.

    Const:
        epsilon (float): When predicting a value close to the border of the interval (so in the case of E_min or E_max) the prediction will overflow to the over range of the interval decreasing the visibility of the result. To fix that we add a shift to the value of E_max and E_min. Epsilon lead to a decrease of resolution on the result, thus we want to choose it as small as possible. If epsilon is too small we won't be able to read the result.
        t_max (float): Is a scalling factor.

    Returns:
        U (quantumCircuit): The whole circuit for the QPE methode.
        l_proba (list): List of the proba associated to each of l_key state (sum should add to one)
        l_lambda (list of float): list that contain the eigenvalue of H. It has the same order has l_key and l_proba.
        l_label (list of str): Give the form of the actual output of the state in binary. (list of ["10100011","01110110"] for example
    """
    epsilon=0.5

    E_max=E_max+epsilon
    E_min=E_min-epsilon

    t_max=1
    print("t_max is :",t_max)

    U=assemble_circuit_for_QPE(phi0,theta,m,dt,t_max,E_min,E_max)
    l_key,l_proba=simulate_QPE(U,nb_shot,m)
    l_proba,l_lambda,l_label=treatment_result_QPE(l_key,l_proba,m,E_min,E_max,t_max)

    return U,l_proba,l_lambda,l_label

def plot_m_2_4_6(phi0,theta,dt,nb_shot,E_max,E_min):
    """This function is intended to plot the result on a single plot on four different figure corresponding to different value of m the number of ancillary qubit to showcase the impact of it.
    It call the main_QPE function with different arguments.

    Args:
        phi0 (list of int): List of 0 and 1 describing the initial state of the circuit
        theta(float): Mixing angle for a specific 2 flavor case
        m (int): Number of ancillary qubit, qubit that hold the eingenvalue prediction in our case.
        dt (float): value of a time step
        nb_shot (int): Sampling parameter that indicate how many time we run the circuit. A too low value may lead significative statistical fluctuation.
        E_min (float): minimal value of the spectrum of H. We don't need the exact value but a rough estimate of it's value, anything smaller than E_min will work but too small value will decrease the resolution.
        E_max (float): maximal value of the spectrum of H. We don't need the exact value but a rough estimate of it's value, anything bigger thatn E_max will work but too big value will decrease the resolution.

    Const:
        L (list of int): This list hold the different value of m to use for each plot.

    Returns:
        void
        plot composed of four figure for different value of m
    """

    L_l_proba=[]
    L_l_lambda=[]
    L_l_label=[]

    L=[2,4,6,8]
    for i in L:
        U2,l_proba2,l_lambda2,l_label2=main_QPE(phi0,theta,i,dt,nb_shot,E_max,E_min)
        L_l_proba.append(l_proba2)
        L_l_lambda.append(l_lambda2)
        L_l_label.append(l_label2)

    fig, axs = plt.subplots(len(L), 1, figsize=(6, 8))
    for i in range(len(L)):
        axs[i].bar(L_l_lambda[i],L_l_proba[i],label="m="+str(L[i]))
        axs[i].tick_params(axis="x", labelsize=14)
        axs[i].tick_params(axis="y", labelsize=14)
        axs[i].set_ylabel("Probability", fontsize=18)
        axs[i].set_xticks([k for k in range(0,len(L_l_lambda[i]),int(np.ceil(len(L_l_lambda[i])/4)))])
        #axs[i].text( 75, 0.50, "m="+str(L[i]),fontsize=18)
        axs[i].legend(bbox_to_anchor=(1, 1),loc="upper right")
    axs[-1].set_xlabel("Energy", fontsize=18)
    plt.show()

##loading and saving
"""This section focused on loading and saving the result of the different simulation.
These function are link to an old way of storing. Later i change the way of storing data.
"""

def load_Zi_l_t_classical(dim,path):
    """This function load the classical result as a reference. It load the operator Z.

    Args:
        dim (int): Number of qubit/Number of neutrino we are using
        path (str): Path where the classical results of the two flavor oscillation are are stored

    Returns:
        L_t (list of float): List of time point used for the classical calculation
        Zi_t (array of float): Zi_t[t,i] give the value of the Z for neutrino i at time t.
    """
    print("loading Zi_t theoric classical solution:")
    with open(path+"saved_Zit_dim_"+str(dim)+".npy", 'rb') as f:
        Zi_t=np.load(f)
    with open(path+"saved_tz_dim_"+str(dim)+".npy", 'rb') as f:
        L_t=np.load(f)
    print("loading Zi_t theoric classical solution done")
    return L_t,Zi_t

def saving_tab_result_quantum(tab_result,l_state_order,l_t,dt,nb_shot,path):
    """The function save the observable Z using the old way of storing data.

    Args:
        tab_result (array of shape (2**len(phi0),k_step)): Array storing the occurence of each state for all the diffent time. tab_result(i,t) will give the occurence of the state id i at the time t
        l_state (list of (str of len N)): List of the the state corresponding to each index of the tab_result array
        l_t (list of float): List of time point used for the quantum emulation
        dt (float): value of a time step
        nb_shot (int): Sampling parameter that indicate how many time we run the circuit. A too low value may lead significative statistical fluctuation.
        path (str): Path to which we store the quantum result (specificaly the old format of quantum result)
    """
    print("save tab_result_sampling quantum")
    nb_step=len(l_t)
    print(np.shape(tab_result))
    dim=len(l_state_order[0])
    with open(path+"V2_saved_tab_res_sampling_dim_"+str(dim)+"_quantum_"+str(nb_step)+"_"+str(dt)+"_"+str(nb_shot)+".npy", 'wb') as f:
        np.save(f, tab_result)
    with open(path+"V2_saved_l_state_order_dim_"+str(dim)+"_quantum_"+str(nb_step)+"_"+str(dt)+"_"+str(nb_shot)+".npy", 'wb') as f:
        np.save(f, l_state_order)
    with open(path+"V2_saved_tz_dim_"+str(dim)+"_quantum_"+str(nb_step)+"_"+str(dt)+"_"+str(nb_shot)+".npy", 'wb') as f:
        np.save(f, np.array(l_t))
    print("Zi_t quantum is saved")

def loading_saving_tab_result_quantum(nb_step,dim,dt,nb_shot,path):
    """The function load the observable Z stored in the old way of storing data.

    Args:
        nb_step (float): number of time point we have
        dim (int): Number of qubit/Number of neutrino we are using
        dt (float): value of a time step
        nb_shot (int): Sampling parameter that indicate how many time we run the circuit. A too low value may lead significative statistical fluctuation.
        path (str): Path to which we store the quantum result (specificaly the old format of quantum result)
    """
    print("loading tab_result_sampling quantum")
    with open(path+"V2_saved_tab_res_sampling_dim_"+str(dim)+"_quantum_"+str(nb_step)+"_"+str(dt)+"_"+str(nb_shot)+".npy", 'rb') as f:
        tab_result=np.load(f)
    with open(path+"V2_saved_l_state_order_dim_"+str(dim)+"_quantum_"+str(nb_step)+"_"+str(dt)+"_"+str(nb_shot)+".npy", 'rb') as f:
        l_state_order=np.load(f)
    with open(path+"V2_saved_tz_dim_"+str(dim)+"_quantum_"+str(nb_step)+"_"+str(dt)+"_"+str(nb_shot)+".npy", 'rb') as f:
        l_t=np.load(f)

    print("tab_result_sampling quantum is loaded")
    return tab_result,l_state_order,l_t


##New format calculation and storing:
"""In this section, the goal was to compute some quantity in order to compare to other methods. For that we also change the formating of the data. You find all the function related to this new formating even the ploting function.
"""

def concatenate_t_Z(tab_Z,l_temps):
    """This function is the function that change the format of date. It exange the place of the index i and t. Then it add one more collumns we place at the begining correspond to the time point.

    Args:
        tab_Z (array): Zi_t[i,t] give the value of the Z for neutrino i at time t. Result of the Z operator in the old format.
        l_temps (list of float): List of time point used in the calculation

    Returns:
        b (array): b[idt,i+1] give the value of the Z for neutrino i at time the time step idt. Result are given in the new format. And b[idt,0] give the time corresponding at the time step idt
    """
    tab=np.copy(tab_Z)
    a=np.array(list(zip(*tab[::-1]))) #rotate the table by -pi/2 change collumns to ligne here Zn-1 Zn-2 ... Z0
    b=np.c_[l_temps,a[::,::-1]] #add the time and put the correct order of Zi, here temps, Z0 Z1 ... Zn
    return b


def compute_approximate_one_entropy(l_temps,tab_Z):
    """This function compute the approximate average entropy.

    Args:
        l_temps (list of float): List of time point used in the calculation
        tab_Z (array): tab_Z[idt,i+1] give the value of the Z for neutrino i at time the time step idt. Result are given in the new format. And tab_Z[idt,0] give the time corresponding at the time step idt

    Result:
        tab_out (array): Array where the first column correspond to the time point and the second collumns store the average approximate entropy. tab_out[idt,0] give the time corresponding to the time step idt. tab_out[idt,1] give the average approximate entropy at the time step idt.
    """

    nb_temps,n_neutrino=np.shape(tab_Z)
    n_neutrino=n_neutrino
    tab_out=np.zeros((nb_temps,2))
    tab_out[:,0]=l_temps

    for t in range(nb_temps):
        out=0
        for i in range(1,n_neutrino):
            p0=(1/2)*(1+tab_Z[t,i])
            p1=(1/2)*(1-tab_Z[t,i])
            out=out+p0*np.log2(p0)+p1*np.log2(p1)
        tab_out[t,1]=-out/n_neutrino
    return tab_out

def compute_Czz_Cdzz_Codzz(tab_Z,tab_Zi_Zj,l_temps):
    """This function compute the fluctuation term of the Z operator.

    Args:
        tab_Z (array): :Zi_t[i,t] give the value of the Z for neutrino i at time t. Result of the Z operator in the old format.
        tab_Zi_Zj (array of shape (nb_qubit,nbqubit,nbt): array storing the Zij(t) result. tab_out[i,j,idt] give the value of the Zij operator at the time step idt.
        l_temps (list of float): List of time point used in the calculation

    Returns:
        tab_out (array of shape (nbt,4)): Array that all the result. The first column tab_out[:,0] store the time, the second column store tab_out[:,1] store the czz measure, the third column tab_out[:,2] store the cdzz, the third columns store the codzz.
    """
    print(np.shape(tab_Z))
    n_neutrino,nb_temps=np.shape(tab_Z)
    tab_out=np.zeros((nb_temps,4))
    tab_out[:,0]=l_temps


    for t in range(nb_temps):
        czz=0
        cdzz=0
        codzz=0
        for i in range(n_neutrino):
            cdzz=cdzz+tab_Zi_Zj[i,i,t]-(tab_Z[i,t])**2
            for j in range(n_neutrino):
                czz=czz+tab_Zi_Zj[i,j,t]-tab_Z[i,t]*tab_Z[j,t]
                if i!=j:
                    codzz=codzz+tab_Zi_Zj[i,j,t]-tab_Z[i,t]*tab_Z[j,t]
        tab_out[t,1]=czz/(n_neutrino**2)
        tab_out[t,2]=cdzz/(n_neutrino**2)
        tab_out[t,3]=codzz/(n_neutrino**2)
    return tab_out


def compute_entropy(tab_X,tab_Y,tab_Z,l_temps):
    """This function compute the exact entropy for each neutrino and the average entropy on the neutrino using the X,Y and Z measurement.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    We result of X and Y are not trusworthy see add_measurment function for more information
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Args:
        tab_X (array): tab_X[idt,i+1] give the value of the X for neutrino i at time the time step idt. Result are given in the new format. And tab_X[idt,0] give the time corresponding at the time step idt. This is the result that is not trustworthy.
        tab_Y (array): tab_Y[idt,i+1] give the value of the Y for neutrino i at time the time step idt. Result are given in the new format. And tab_Y[idt,0] give the time corresponding at the time step idt. This is the result that is not trustworthy.
        tab_Z (array): tab_Z[idt,i+1] give the value of the Z for neutrino i at time the time step idt. Result are given in the new format. And tab_Z[idt,0] give the time corresponding at the time step idt
        l_temps (list of float): List of time point used in the calculation

    Returns:
        tab_entropy_f (array): Contain the exact entropy. tab_entropy_f[:,0] give you the time point and tab_entropy_f[:,i] with i in [1,nb_neutrino] give you the entropy of a specific neutrino.
        tab_av_entropy_f (array): Contain the average exact entropy (in opposition to the average approximate entropy). tab_av_entropy_f[:,0] give you the time point and tab_av_entropy_f[:,1] give you the actual value of the average exact entropy.
    """
    l_densite=[]

    #for i in range(len(tab_Z)):
    for i in range(1,len(tab_Z[0])):#alternative version in the case of tab_Z_final

        l=[]

        #for t in range(len(tab_Z[0])):
        for t in range(len(tab_Z)):#alternative version in the case of tab_Z_final

            D=1/2*np.matrix([[1+tab_Z[t,i],tab_X[t,i]+1j*tab_Y[t,i]],[np.conjugate(tab_X[t,i]+1j*tab_Y[t,i]),1-tab_Z[t,i]]])
            eig=np.linalg.eig(D)[0]
            s=-eig[0]*np.log2(eig[0])-eig[1]*np.log2(eig[1])
            l.append(s)
        l_densite.append(l)

    l_densite=np.real(l_densite)

    l_av_entropy=[]
    for t in range(len(l_densite[0])):
        s=0
        for i in range(len(l_densite)):
            s=s+l_densite[i][t]
        l_av_entropy.append(s/(len(l_densite)))

    nb_temps=len(l_temps)
    nb_neutrino,nb_temps=np.shape(l_densite)


    tab_entropy_f=np.zeros((nb_temps,nb_neutrino+1))
    tab_entropy_f[:,0]=l_temps

    for i in range(nb_neutrino):
        tab_entropy_f[:,i+1]=l_densite[i]

    tab_av_entropy_f=np.zeros((nb_temps,2))
    tab_av_entropy_f[:,0]=l_temps
    tab_av_entropy_f[:,1]=l_av_entropy

    return tab_entropy_f,tab_av_entropy_f


def save_date_new_format(tab_t_Z=[], tab_Czz_Cdzz_Codzz_f=[], tab_approximate_entropy_f=[], tab_entropy_f=[], tab_entropy_average_f=[],path="",tab_t_X=[],tab_t_Y=[]):
    """This fonction save all the calculation done in the new format on a file.

    Args:
        tab_t_Z (array): tab_t_Z[idt,i+1] give the value of the Z operator for neutrino i at time the time step idt. Result are given in the new format. And tab_t_Z[idt,0] give the time corresponding at the time step idt
        tab_Czz_Cdzz_Codzz_f (array): tab_Czz_Cdzz_Codzz_f (array of shape (nbt,4)): Array that all the result. The first column tab_Czz_Cdzz_Codzz_f[:,0] store the time, the second column store tab_Czz_Cdzz_Codzz_f[:,1] store the czz measure, the third column tab_Czz_Cdzz_Codzz_f[:,2] store the cdzz, the third columns store the codzz.
        tab_approximate_entropy_f (array): Array where the first column correspond to the time point and the second collumns store the average approximate entropy. tab_approximate_entropy_f[idt,0] give the time corresponding to the time step idt. tab_approximate_entropy_f[idt,1] give the average approximate entropy at the time step idt.
        tab_entropy_f (array): Contain the exact entropy. tab_entropy_f[:,0] give you the time point and tab_entropy_f[:,i] with i in [1,nb_neutrino] give you the entropy of a specific neutrino.
        tab_av_entropy_f (array): Contain the average exact entropy (in opposition to the average approximate entropy). tab_av_entropy_f[:,0] give you the time point and tab_av_entropy_f[:,1] give you the actual value of the average exact entropy.
        path (str): path to which we will save the data
        tab_t_X (array): tab_t_X[idt,i+1] give the value of the X operator for neutrino i at time the time step idt. Result are given in the new format. And tab_t_X[idt,0] give the time corresponding at the time step idt. This is the result that is not trustworthy.
        tab_t_Y (array): tab_t_Y[idt,i+1] give the value of the Y operator for neutrino i at time the time step idt. Result are given in the new format. And tab_t_Y[idt,0] give the time corresponding at the time step idt. This is the result that is not trustworthy.

    Returns:
        void
        Create multiple file to save the data to the file path given by "path"
    """

    n_temps,n_neutrino=np.shape(tab_t_Z)
    n_neutrino=n_neutrino-1
    print(n_neutrino)

    if len(tab_t_Z)!=0:
        np.savetxt(path+"n="+str(n_neutrino)+"_Zi_per_neutrino.dat",tab_t_Z)

    if len(tab_t_X)!=0:
        np.savetxt(path+"n="+str(n_neutrino)+"_Xi_per_neutrino.dat",tab_t_X)

    if len(tab_t_Y)!=0:
        np.savetxt(path+"n="+str(n_neutrino)+"_Yi_per_neutrino.dat",tab_t_Y)

    if len(tab_Czz_Cdzz_Codzz_f) !=0:
        np.savetxt(path+"n="+str(n_neutrino)+"_Czz_Cdzz_Codzz.dat",tab_Czz_Cdzz_Codzz_f)

    if len(tab_approximate_entropy_f) !=0:
        np.savetxt(path+"n="+str(n_neutrino)+ "_Entropy_average_approximate.dat",tab_approximate_entropy_f)

    #np.savetxt(path+"n="+str(n_neutrino)+"_Exact_entropy_per_neutrino.dat",tab_entropy_f)

    if len(tab_entropy_average_f)!=0:
        np.savetxt(path+"n="+str(n_neutrino)+"_Exact_average_entropy_per_neutrino.dat",tab_entropy_average_f)


def load_data_new_format(n_neutrino,path):
    """This fonction load the data that has been save using save_date_new_format. It's loading both the classical result and the quantum result.

    Args:
        n_neutrino (int): number of neutrino
        path (str): path from which we want to load the data from

    Returns:
        Zi_t_c (array): Zi_t_c[idt,i+1] give the value of the X for neutrino i for the classical computation at the time step idt. Result are given in the new format. And Zi_t_c[idt,0] give the time corresponding at the time step idt
        Xi_t_c (array): Xi_t_c[idt,i+1] give the value of the X for neutrino i for the classical computation at the time step idt. Result are given in the new format. And Xi_t_c[idt,0] give the time corresponding at the time step idt
        Yi_t_c (array): Yi_t_c[idt,i+1] give the value of the X for neutrino i for the classical computation at the time step idt. Result are given in the new format. And Yi_t_c[idt,0] give the time corresponding at the time step idt
        tab_Czz_Cdzz_Codzz_c (array of shape (nbt,4)): Array that contain all the result related to fluctuation from the classical computation. The first column tab_Czz_Cdzz_Codzz_c[:,0] store the time, the second column store tab_Czz_Cdzz_Codzz_c[:,1] store the czz measure, the third column tab_Czz_Cdzz_Codzz_c[:,2] store the cdzz, the third columns store the codzz.
        tab_approximate_entropy_average_c (array): Array where the first column correspond to the time point and the second collumns store the average approximate entropy. tab_approximate_entropy_f[idt,0] give the time corresponding to the time step idt. tab_approximate_entropy_f[idt,1] give the average approximate entropy at the time step idt.
        tab_entropy_average_c (array): Contain the average exact entropy (in opposition to the average approximate entropy). tab_av_entropy_f[:,0] give you the time point and tab_av_entropy_f[:,1] give you the actual value of the average exact entropy.
        Zi_t_q (array): Zi_t_q[idt,i+1] give the value of the Z for neutrino i at time the time step idt. Result are given in the new format. And Zi_t_q[idt,0] give the time corresponding at the time step idt
        tab_Czz_Cdzz_Codzz_q (array of shape (nbt,4)): Array that contain all the result related to fluctuation from the quantum computation. The first column tab_Czz_Cdzz_Codzz_q[:,0] store the time, the second column store tab_Czz_Cdzz_Codzz_q[:,1] store the czz measure, the third column tab_Czz_Cdzz_Codzz_q[:,2] store the cdzz, the third columns store the codzz.
        tab_approximate_entropy_average_q (array): Array where the first column correspond to the time point and the second collumns store the average approximate entropy. tab_approximate_entropy_average_q[idt,0] give the time corresponding to the time step idt. tab_approximate_entropy_average_q[idt,1] give the average approximate entropy at the time step idt.
        tab_entropy_average_q (array): Contain the average exact entropy (in opposition to the average approximate entropy). tab_entropy_average_q[:,0] give you the time point and tab_entropy_average_q[:,1] give you the actual value of the average exact entropy.
    """

    with open(path+"n="+str(n_neutrino)+"_Zi_per_neutrino.dat", 'rb') as f:
        Zi_t_q=np.loadtxt(f)

    with open(path+"n="+str(n_neutrino)+"_Czz_Cdzz_Codzz.dat", 'rb') as f:
        tab_Czz_Cdzz_Codzz_q=np.loadtxt(f)
    with open(path+"n="+str(n_neutrino)+ "_Entropy_average_approximate.dat", 'rb') as f:
        tab_approximate_entropy_average_q=np.loadtxt(f)
    with open(path+"n="+str(n_neutrino)+"_Exact_average_entropy_per_neutrino.dat", 'rb') as f:
        tab_entropy_average_q=np.loadtxt(f)

    with open(path+"n="+str(n_neutrino)+"_classical_Zi_per_neutrino.dat", 'rb') as f:
        Zi_t_c=np.loadtxt(f)
    with open(path+"n="+str(n_neutrino)+"_classical_Xi_per_neutrino.dat", 'rb') as f:
        Xi_t_c=np.loadtxt(f)
    with open(path+"n="+str(n_neutrino)+"_classical_Yi_per_neutrino.dat", 'rb') as f:
        Yi_t_c=np.loadtxt(f)

    with open(path+"n="+str(n_neutrino)+"_classical_Czz_Cdzz_Codzz.dat", 'rb') as f:
        tab_Czz_Cdzz_Codzz_c=np.loadtxt(f)
    with open(path+"n="+str(n_neutrino)+ "_classical_Entropy_average_approximate.dat", 'rb') as f:
        tab_approximate_entropy_average_c=np.loadtxt(f)
    with open(path+"n="+str(n_neutrino)+"_classical_Exact_average_entropy_per_neutrino.dat", 'rb') as f:
        tab_entropy_average_c=np.loadtxt(f)

    return Zi_t_c,Xi_t_c,Yi_t_c,tab_Czz_Cdzz_Codzz_c,tab_approximate_entropy_average_c,tab_entropy_average_c,Zi_t_q,tab_Czz_Cdzz_Codzz_q,tab_approximate_entropy_average_q,tab_entropy_average_q

def load_data_new_format_operator_only(n_neutrino,path):
    """This function load only the quantum result of the Z, X and Y operator. X and Y are not trustworthy.

    Args:
        n_neutrino (int): number of neutrino we work with
        path (str): path to which the data are saved to

    Returns:
        Xi_t_q (array): Xi_t_q[idt,i+1] give the value of the X operator for neutrino i at time the time step idt. Result are given in the new format. And Xi_t_q[idt,0] give the time corresponding at the time step idt
        Yi_t_q (array): Yi_t_q[idt,i+1] give the value of the Y operator for neutrino i at time the time step idt. Result are given in the new format. And Yi_t_q[idt,0] give the time corresponding at the time step idt
        Zi_t_q (array): Zi_t_q[idt,i+1] give the value of the Z operator for neutrino i at time the time step idt. Result are given in the new format. And Zi_t_q[idt,0] give the time corresponding at the time step idt
    """

    with open(path+"n="+str(n_neutrino)+"_Zi_per_neutrino.dat", 'rb') as f:
        Zi_t_q=np.loadtxt(f)
    with open(path+"n="+str(n_neutrino)+"_Xi_per_neutrino.dat", 'rb') as f:
        Xi_t_q=np.loadtxt(f)
    with open(path+"n="+str(n_neutrino)+"_Yi_per_neutrino.dat", 'rb') as f:
        Yi_t_q=np.loadtxt(f)

    return Xi_t_q,Yi_t_q,Zi_t_q

def plot_comparaison_classical_quantum_Zi_t(N):
    """This fonction load the Z operator for both classical and quantum computation and plot the time propagation for every neutrino in a figure.

    Args:
        N (int): number of neutrino to consider

    Returns:
        void
        plot the comparaison figure between the classical and quantum computation of the Z operator
    """

    Zi_t_c,Xi_t_c,Yi_t_c,tab_Czz_Cdzz_Codzz_c,tab_approximate_entropy_average_c,tab_entropy_average_c,Zi_t_q,tab_Czz_Cdzz_Codzz_q,tab_approximate_entropy_average_q,tab_entropy_average_q=load_data_new_format(N,"/Users/bauge/Desktop/stage RP/recherche/data_for_denis/")

    n_dt,n_neutrino=np.shape(Zi_t_c)
    n_neutrino=n_neutrino-1

    l_color=list(mcolors.TABLEAU_COLORS.values())#color up to 10 differents color, else we need to use something else
    l_color=l_color+list(matplotlib.colors.cnames.values())
    l_temps=Zi_t_c[:,0]
    for i in range(0,n_neutrino):
        plt.plot(l_temps,Zi_t_c[:,i+1],color=l_color[i],label="n="+str(i+1)+" classical")
    for i in range(0,n_neutrino):
        plt.plot(l_temps,Zi_t_q[:,i+1],color=l_color[i],label="n="+str(i+1)+" quantum",linestyle="None",marker=".",markersize=5)

    plt.xlabel("Time $[µ^{-1}]$", fontsize=18)
    plt.ylabel("$<Z_i>(t)$", fontsize=18)
    plt.legend()
    plt.show()

def plot_comparaison_classical_quantum_fluctuation(N):
    """This function plot the classical vs quantum comparaison of the fluctuation.

    Args:
        N (int): number of neutrino to consider

    Returns:
        void
        plot the comparaison figure between the classical and quantum computation of the fluctuation of the Z operator.
    """

    Zi_t_c,Xi_t_c,Yi_t_c,tab_Czz_Cdzz_Codzz_c,tab_approximate_entropy_average_c,tab_entropy_average_c,Zi_t_q,tab_Czz_Cdzz_Codzz_q,tab_approximate_entropy_average_q,tab_entropy_average_q=load_data_new_format(N,"/Users/bauge/Desktop/stage RP/recherche/data_for_denis/")

    n_dt,n_courbe=np.shape(tab_Czz_Cdzz_Codzz_c)
    print(np.shape(tab_Czz_Cdzz_Codzz_c))
    n_courbe=n_courbe-1

    l_label=["Czz","Cdzz","Codzz"]

    l_color=list(mcolors.TABLEAU_COLORS.values())#color up to 10 differents color, else we need to use something else
    l_color=l_color+list(matplotlib.colors.cnames.values())
    l_temps=tab_Czz_Cdzz_Codzz_c[:,0]
    for i in range(0,n_courbe):
        plt.plot(l_temps,tab_Czz_Cdzz_Codzz_c[:,i+1],color=l_color[i],label=l_label[i]+" classical")
    for i in range(0,n_courbe):
        plt.plot(l_temps,tab_Czz_Cdzz_Codzz_q[:,i+1],color=l_color[n_courbe+i],label=l_label[i]+" quantum",linestyle="None",marker=".",markersize=3)


    plt.xlabel("Time $[µ^{-1}]$", fontsize=18)
    plt.ylabel("fluctuation", fontsize=18)
    plt.legend()
    plt.show()

def plot_comparaison_classical_quantum_entropy_average(N):
    """This function plot the comparaison of the average exact entropy between the classical and the quantum computation.
    !!!!!!!!!!!!!!!!!!!!!!!!
    The quantum result are not 100% trustworthy
    !!!!!!!!!!!!!!!!!!!!!!!!

    Args:
        N (int): number of neutrino to consider

    Returns:
        void
        plot the comparaison figure between the classical and quantum computation of the average exact entropy.
    """

    Zi_t_c,Xi_t_c,Yi_t_c,tab_Czz_Cdzz_Codzz_c,tab_approximate_entropy_average_c,tab_entropy_average_c,Zi_t_q,tab_Czz_Cdzz_Codzz_q,tab_approximate_entropy_average_q,tab_entropy_average_q=load_data_new_format(N,"/Users/bauge/Desktop/stage RP/recherche/data_for_denis/")

    n_dt,a=np.shape(tab_approximate_entropy_average_c)
    print(np.shape(tab_approximate_entropy_average_c))

    l_label=["app. av. classique","app. av. quantique","av. classique","av. quantique"]

    l_color=list(mcolors.TABLEAU_COLORS.values())#color up to 10 differents color, else we need to use something else
    l_color=l_color+list(matplotlib.colors.cnames.values())
    l_temps=tab_approximate_entropy_average_c[:,0]

    plt.plot(l_temps,tab_entropy_average_c[:,1],color=l_color[2],label=l_label[2])
    plt.plot(l_temps,tab_approximate_entropy_average_c[:,1],color=l_color[0],label=l_label[0])
    plt.plot(l_temps,tab_entropy_average_q[:,1],color=l_color[3],label=l_label[3],linestyle="None",marker=".",markersize=3)
    plt.plot(l_temps,tab_approximate_entropy_average_q[:,1],color=l_color[1],label=l_label[1],linestyle="None",marker=".",markersize=3)
    plt.xlabel("Time $[µ^{-1}]$", fontsize=18)
    plt.ylabel("entropy", fontsize=18)
    plt.legend()
    plt.show()




##print/plot function
"""These function regroupe the rest of the function, they are print or plot function that served as a visualisation or as verification of the result. They contain the remaining plot function.
"""

def print_circuit(circuit):
    """Function that plot a quantum circuit."""
    circuit.draw(output="mpl")
    plt.show()

def plot_z_state(L_t, tab_Z_ti):
    """This function plot the evolution of the operator Z over time for each neutrino.

    Args:
        L_t (list): list of time point
        tab_Z_ti (array): Zi_t[i,t] give the value of the Z for neutrino i at time t. Result of the Z operator in the old format.

    Returns:
        void
        plot the corresponding graph
    """
    n,ndt=np.shape(tab_Z_ti)
    for i in range(n):
        plt.plot(L_t,tab_Z_ti[i,:],label="n="+str(i)+" quantum")
    plt.xlabel("Time $[µ^{-1}]$", fontsize=18)
    plt.ylabel("$<Z_i>(t)$", fontsize=18)
    plt.legend()
    plt.show()

def plot_test():
    """This function make two different plot.
    The first plot is composed of 4 figure of the evolution of the Z operator for multiple neutrino given different time step dt. It show the impact of the choice of the time step.
    The second plot is composed of 4 figure of the evolution of the Z operator for multiple neutrino given different number of shot. It illustrate the importance of statistical error.
    """
    nb_shot=10000

    theta=0.195
    occ=1000
    dt=0.1


    L_tab_Z1=[]
    L_l_temps1=[]

    L_tab_Z2=[]
    L_l_temps2=[]

    l_dt=[0.1,0.5,1,5]
    l_occ=[1000,200,100,21]
    l_nb_shot=[10000,1000,100]
    n_neutrino=4

    L_t_classical,Zi_t_classical=load_Zi_l_t_classical(n_neutrino,path_saved_file)

    for i in range(len(l_dt)): #variying only dt with default parameter otherwise
        tab_result1,l_state_order1,L_temps1=loading_saving_tab_result_quantum(l_occ[i],len(phi_init),l_dt[i],nb_shot,path_saved_file)
        tab_Z1=evaluate_operator_Z(tab_result1,l_state_order1)
        L_tab_Z1.append(tab_Z1)
        L_l_temps1.append(L_temps1)


    fig, axs = plt.subplots(len(l_dt), 1, figsize=(6, 8), sharex=True)
    l_color=list(mcolors.TABLEAU_COLORS.values())
    for i in range(len(l_dt)):
        axs[i].plot(L_l_temps1[i],L_tab_Z1[i][0,:],l_color[0],label="neutrino "+str(1))
        axs[i].plot(L_l_temps1[i],L_tab_Z1[i][2,:],l_color[1],label="neutrino "+str(3))
        axs[i].plot(L_t_classical,Zi_t_classical[0],l_color[2],linestyle=":",label="neutrino "+str(1)+" classical")
        axs[i].plot(L_t_classical,Zi_t_classical[2],l_color[3],linestyle=":",label="neutrino "+str(3)+" classical")
        axs[i].tick_params(axis="x", labelsize=14)
        axs[i].tick_params(axis="y", labelsize=14)
        axs[i].set_ylabel("$<Z_{H_1}>(t)$", fontsize=18)
        axs[i].text( 75, 0.50, "dt="+str(l_dt[i]),fontsize=18)
        axs[i].set_xlim(0, 100)
    axs[1].legend(bbox_to_anchor=(1.10, 0),loc="lower right")
    axs[-1].set_xlabel("Time $[µ^{-1}]$", fontsize=18)
    plt.show()

    for i in l_nb_shot: #variying only dt with default parameter otherwise
        tab_result1,l_state_order1,L_temps1=loading_saving_tab_result_quantum(occ,len(phi_init),dt,i,path_saved_file)
        tab_Z1=evaluate_operator_Z(tab_result1,l_state_order1)
        L_tab_Z2.append(tab_Z1)
        L_l_temps2.append(L_temps1)

    fig, axs = plt.subplots(len(l_nb_shot), 1, figsize=(6, 8), sharex=True)
    l_color=list(mcolors.TABLEAU_COLORS.values())
    for i in range(len(l_nb_shot)):
        axs[i].plot(L_l_temps2[i],L_tab_Z2[i][0,:],l_color[0],label="neutrino "+str(1))
        axs[i].plot(L_l_temps2[i],L_tab_Z2[i][2,:],l_color[1],label="neutrino "+str(3))
        axs[i].plot(L_t_classical,Zi_t_classical[0],l_color[2],linestyle=":",label="neutrino "+str(1)+" classical")
        axs[i].plot(L_t_classical,Zi_t_classical[2],l_color[3],linestyle=":",label="neutrino "+str(3)+" classical")
        axs[i].tick_params(axis="x", labelsize=14)
        axs[i].tick_params(axis="y", labelsize=14)
        axs[i].set_ylabel("$<Z_{H_1}>(t)$", fontsize=18)
        axs[i].text( 75, 0.50, "nb_shot="+str(l_nb_shot[i]),fontsize=18)
        axs[i].set_xlim(0, 100)
    axs[1].legend(bbox_to_anchor=(1.10, 0),loc="lower right")
    axs[-1].set_xlabel("Time $[µ^{-1}]$", fontsize=18)
    plt.show()

def plot_dt_error_interraction_terme():
    """This function compute the error we do for one time step for different version of the propagator and different size of step. It plot the erroer of each methode in function of the size of the time step.
    """
    def exact_solution(dt):
        """This function compute the reference result that we aim to obtain"""
        N=2
        jij=Jij(N)
        SigmaX=np.matrix([[0,1],[1,0]])
        SigmaY=np.matrix([[0,-1j],[ 1j,0]])
        SigmaZ=np.matrix([[1,0],[0,-1]])
        return expm(-1j*dt*jij[0,1]*(np.kron(SigmaX,SigmaX)+np.kron(SigmaY,SigmaY)+np.kron(SigmaZ,SigmaZ)))
    def c_error(M1,M2):
        """This function sum the absolute difference of two matrices"""
        return(sum(sum(np.abs(M1-M2))))
    def l_c_error(l_M1,l_M2):
        """construct the list of the error in function of the time step size"""
        l=[]
        for i in range(len(l_M1)):
            l.append(c_error(l_M1[i],l_M2[i]))
        return l
    N=2
    theta=0.195
    l_dt=np.logspace(-5,5,100)
    print(l_dt)
    L_exact=[exact_solution(dt) for dt in l_dt]



    L_U_full_trotter=[qi.Operator(U2_propagator_1_step(dt,N)) for dt in l_dt]
    L_U_smaller_trotter=[qi.Operator(U2_propagator_1_step_v2(dt,N)) for dt in l_dt]
    L_U_exact=[qi.Operator(U2_propagator_1_step_optimal(dt,N)) for dt in l_dt]

    plt.plot(l_dt,l_c_error(L_exact,L_U_full_trotter),label="Error double Trotter")
    plt.plot(l_dt,l_c_error(L_exact,L_U_smaller_trotter),label="Error XX+YY gate Trotter")
    plt.plot(l_dt,l_c_error(L_exact,L_U_exact),label="Error optimal")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("dt $[µ^{-1}]$", fontsize=18)
    plt.ylabel("maximal absolute error", fontsize=18)
    plt.legend()
    plt.show()

def plot_error_trotter_interraction(n):
    """This function compute the trotter error as described by Amitrano in her article (See top of the file).
    It then plot this trotter error in function of the time step"""
    def error(N):
        jij=Jij(N)
        print(jij)
        error_wo_fac=0
        for i in range(N):
            for j in range(i+1,N):
                s1=0
                s2=0
                for l in range(j+1,N):
                    s1=s1+jij[i,l]-jij[j,l]
                for k in range(i+1,N):
                    s2=s2+jij[k,j]
                error_wo_fac=error_wo_fac+jij[i,j]*np.abs(s1+s2)
        return(error_wo_fac)

    l_dt=np.logspace(-5,5,100)
    for j in range(3,n+1):
        error_j=error(j)
        l_error=[((l_dt[i])**2)*4*error_j for i in range(len(l_dt))]
        plt.plot(l_dt,l_error,label="U2 trotter error for $u_{ij}$ exact for "+str(j)+" neutrino")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("dt $[µ^{-1}]$", fontsize=18)
    plt.ylabel("upper bound error", fontsize=18)
    plt.legend()
    plt.show()

def plot_choose_best_configuration(bool_case_quantique=False):
    """This function compare the evolution of the number of one and two qubit gate in the case of quantum emulation or on a quantum computer for different version of the propagator (both the case of U1 and U2.
    """

    l_n=[n for n in range(1,31)]

    """The following 4 function give the function that give the evoution of the number of gate for all the different propagator.
    """
    def H1_exact_func(N):
        return 3*N
    def H1_trotter_func(N):
        return 2*N
    def H2_trotter_func(N,bool_case_quantique):
        if bool_case_quantique:
            return N*(N-1)/2*np.array([6,15])
        else:
            return N*(N-1)/2*np.array([3,0])
    def H2_optimal_func(N,bool_case_quantique):
        return N*(N-1)/2*np.array([3,7])

    H1_exact=[H1_exact_func(i) for i in l_n]
    H1_trotter=[H1_trotter_func(i) for i in l_n]

    H2_trotter_c=[H2_trotter_func(i,bool_case_quantique)[0] for i in l_n]
    H2_optimal_c=[H2_optimal_func(i,bool_case_quantique)[0] for i in l_n]
    H2_trotter=[H2_trotter_func(i,bool_case_quantique)[1] for i in l_n]
    H2_optimal=[H2_optimal_func(i,bool_case_quantique)[1] for i in l_n]

    l_color=list(mcolors.TABLEAU_COLORS.values())
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    axs[0].plot(l_n,H1_exact,l_color[0],linestyle="-",label="H1_exact")
    axs[0].plot(l_n,H1_trotter,l_color[1],linestyle="-",label="H1_trotter")

    axs[1].plot(l_n,H2_trotter_c,l_color[0],linestyle="-",label="H2_trotter_2qb")
    axs[1].plot(l_n,H2_trotter,l_color[0],linestyle=":",label="H2_trotter_1qb")

    axs[1].plot(l_n,H2_optimal_c,l_color[1],linestyle="-",label="H2_optimal_2qb")
    axs[1].plot(l_n,H2_optimal,l_color[1],linestyle=":",label="H2_optimal_1qb")

    axs[-1].set_xlabel("number of neutrino", fontsize=18)
    axs[0].set_ylabel("Number of 1 qubit gate")
    axs[1].set_ylabel("Number of gate")
    axs[0].legend()
    axs[1].legend()

    plt.show()

##Application
"""This section contain some already prepared code to directly use the function presented ealier.

Before executing the code, you need to check the path at which data will save and load from:
    - The "path_saved_file" is the location of both the classical result and the old format of the quantum calculation.
    - The "path_data_new_format" is the location of the new format for the quantum computation.

In order to use them you can interract with the following parameter:
    - the initial state is controled by "phi_init" right bellow
    - the number of shot is controled by "nb_shot" right bellow
    - the mixing angle is controled by "theta" right bellow
    - the number of time step to do is controled by "occ" right bellow
    - the time step size is controled by "dt" right bellow
    - which part of the code to run is controlled by the "mode" parameter
    - some other parameter are specific to a mode and will define in the mode explanation

The mode parameter control which task you want to do:

* mode==0 load the old format of data of result and plot it. It plot the time evolution of the operator Z for all the neutrino. We need to run the mode 1 with the same parameter for this fuction to be able to plot the result

* mode==1 do the computation and save the data in the old format of data. It save the data in "path_saved_file"

* mode==2 run the plot_test function

* mode==3 plot the comparaison between the classical and quantum computation of the operator Z. It load the new format of data, so we need to do the calculation by running mode==6

* mode==4 run the plot_dt_error_interraction_terme

* mode==5 run the plot_error_trotter_interraction

* mode==6  this mode do the calculation of everything exept the entropy. The ZZ calculation may take a lot of time. If it taking too long you can still interrupt the process and you will still have everything exept the cdzz and the ZZ. It save the data in "path_data_new_format"
We introduce a new parameter:
    step_mutilple (float): it control how many step we keep for the calculation.

* mode==6.1 this mode do the calculation of the entropy. It save the data in "path_data_new_format"
Warning:
    The computation of the exact entropy is not trust worthy, because of the X and Y measurement.

* mode==7 This mode do the QPE process in order to plot the comparaison graph (see function "plot_m_2_4_6")
For this mode we have to redefine some parameter:
    - phi0 the initial state to study
    - m the number of ancillary qubits
    - dt the time step
    - nb_shot control the number of time to run the circuit
    - E_min and E_max the minimum and maximum eigenvalue of the Hamiltonian ( anything bigger than E_max and smaller than E_min will work). The default parameter are the exact value obtained by diagoanlising the Hamiltonian by hand.

Warning: The function comparing the classical and quantum computation necessitate that you already did the classical computation.

"""

self.path_saved_file="./data_2_flavor/"
self.path_data_new_format="./data_2_flavor_formated_quantum/"

phi_init=[0,0,1,1]

print(phi_init)

nb_shot=10000

theta=0.195

occ=200
dt=1

N=len(phi_init)
L_temps=[dt*i for i in range(occ)]

mode=4

if mode==0:#print mode

    tab_result2,l_state_order2,L_temps2=loading_saving_tab_result_quantum(occ,len(phi_init),dt,nb_shot,path_saved_file)
    tab_Z_t=evaluate_operator_Z(tab_result2,l_state_order2)
    plot_z_state(L_temps2, tab_Z_t)#quantum alone

elif mode==1: #compute mode

    tab_result,l_state_order=simulate_circuit_all_t(occ,dt,theta,N,phi_init,nb_shot)
    tab_Z_t=evaluate_operator_Z(tab_result,l_state_order)
    print(tab_Z_t)
    print(np.shape(tab_Z_t))
    saving_tab_result_quantum(tab_result,l_state_order,L_temps,dt,nb_shot,path_saved_file)

elif mode==2:
    plot_test()

elif mode==3: #load comparaison work for N<10
    plot_comparaison_classical_quantum_Zi_t(N)

elif mode==4:
    plot_dt_error_interraction_terme()

elif mode==5:
    plot_error_trotter_interraction(N)

elif mode==6:
    print(len(phi_init))
    #for this case we need: occ*dt=200 exatly
    #step to keep=4/dt

    #the _f stand for formated, which mean that the output as the intended format to be saved
    step_mutilple=1

    step_to_keep=int(step_mutilple/dt)

    print("creating and simulating the circuit")
    tab_result,l_state_order=simulate_circuit_all_t(occ,dt,theta,N,phi_init,nb_shot)
    tab_result_X,l_state_order_X=simulate_circuit_all_t(occ,dt,theta,N,phi_init,nb_shot,"X")
    tab_result_Y,l_state_order_Y=simulate_circuit_all_t(occ,dt,theta,N,phi_init,nb_shot,"Y")

    print("cutting data off")
    #tronc data at the correct step
    tab_result=tab_result[::,::step_to_keep]
    tab_result_X=tab_result_X[::,::step_to_keep]
    tab_result_Y=tab_result_Y[::,::step_to_keep]
    L_temps=L_temps[::step_to_keep]

    #evaluate Zi and Zi_Zj
    print("calculation of Zi and Zi_Zj")
    tab_Z_t=evaluate_operator_Z(tab_result,l_state_order)
    tab_t_and_Z_f=concatenate_t_Z(tab_Z_t,L_temps)
    save_date_new_format(tab_t_and_Z_f,[],[],[],[], path_data_new_format)
    print("Zi done")

    tab_X_t=evaluate_operator_Z(tab_result_X,l_state_order_X)
    tab_t_and_X_f=concatenate_t_Z(tab_X_t,L_temps)
    save_date_new_format(tab_t_and_Z_f,[],[],[],[], path_data_new_format,tab_t_and_X_f)
    print("Xi done")

    tab_Y_t=evaluate_operator_Z(tab_result_Y,l_state_order_Y)
    tab_t_and_Y_f=concatenate_t_Z(tab_Y_t,L_temps)
    save_date_new_format(tab_t_and_Z_f,[],[],[],[], path_data_new_format,[],tab_t_and_Y_f)
    print("Yi done")


    tab_Zi_Zj=evaluate_operator_Zi_Zj(tab_result,l_state_order)
    print("Zi_Zj done")

    print(np.shape(tab_X_t))
    print(np.shape(tab_Y_t))
    print(np.shape(tab_Z_t))
    print("calculation of Czz,CdZZ,CodZZ")
    #compute Czz,CdZZ,CodZZ
    tab_Czz_Cdzz_Codzz_f=compute_Czz_Cdzz_Codzz(tab_Z_t,tab_Zi_Zj,L_temps)

    print("saving data")
    save_date_new_format(tab_t_and_Z_f,tab_Czz_Cdzz_Codzz_f,[],[],[], path_data_new_format)

elif mode==6.1:
    Xi,Yi,Zi=load_data_new_format_operator_only(len(phi_init),path_data_new_format)
    L_temps=Zi[:,0]

    print("compute approximate entropy")
    tab_approximate_entropy_f=compute_approximate_one_entropy(L_temps,Zi)
    print("computing exact entropy")
    tab_entropy_f,tab_entropy_average_f=compute_entropy(Xi,Yi,Zi,L_temps)
    save_date_new_format(Zi,[],tab_approximate_entropy_f,tab_entropy_f,tab_entropy_average_f, path_data_new_format)

elif mode==7:#QPE
    print("QPE")
    phi0=[0,1]
    m=7
    dt=1

    E_min=-0.95
    E_max=1.05
    =
    """
    E_min=-1.4246794344808942
    E_max=1.5753205655191034
    """
    """
    E_min=-1.8882137771631848
    E_max=2.111786222836816
    """
    nb_shot=10000

    plot_m_2_4_6(phi0,theta,dt,nb_shot,E_max,E_min)



