"""This code is tasked to compute on a quantum emulator (in opposition to classical computation) the three flavor oscillation of neutrino with the two body interraction.
This code regroupe every function and methode related to the quantum emulation of the quantum circuit related to the Hamiltonian introduced by Siwach and al in the following publication: (https://journals.aps.org/prd/abstract/10.1103/PhysRevD.107.023019).
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
from qiskit.circuit.library import UGate
from qiskit.circuit.library import QFT
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library import PhaseGate
from scipy.linalg import expm,logm

#not working properly, you can remove them if you can't import these two
from dask_jobqueue import LSFCluster
from dask.distributed import Client

##
class Circuit:
    """Class that hold a circuit and all the primitive associated to circuit

        Everything is define in the _init_ function which is wrong to do it like this
    """

    def __init__(self,n):
        """This initialisation function will define some parameter and intialise some calculation.

        Args:
            n (int): Number of neutrino to consider in the circuit

        Variables:
            N (int): Number of neutrino to consider in the circuit
            NB_qubits (int): Number of qubits necessary, corresponding to the number of neutrino
            E (float): Energy of each single neutrino we consider in MeV
            delta_m2 (float): Mass difference between the mass state 2 and 1 in kappa^-1. MeV**2
            Delta_m2 (float): Mass difference between the mass state 3 and the mass state 2 in kappa^-1.MeV**2
            wp (float): Frequency small omega for the one body term in kappa^-1.MeV
            Wbp (float): Frequency big omega for the one body term in kappa^-1.MeV
            k (float): Arbitrary unit introduce by siwach, correspond to 19.7 km/MeV
            mu_R_nu (float): main coefficient of the interraction terme
            R_mu (float): Secondary coefficient of the interraction term
            starting_point (float): Shift in the interraction to start from, we can't start from 0 because the interraction would diverge to infinity
            nb_shot (ints): Parameter for the sampler of the quantum emulator to choose the number of sampling to do
            occ (int): Number of step to perform
            dt (float): Size of each step, it needs to be sufficiently small for Trotter to work
            l_t (list of float): List of the time we will do the computation on (no other used, we often compute it back from dt and occ

            path (str): Path to save and load the quantum emulation of the system
            Path_3saveur (str): Path to load the data of classical computation from

            CIRC (QuantumCircuit): The main quantum circuit we will use

        Returns:
            Void
            self
        """

        self.N=n #nb neutrino
        self.NB_qubits=2*self.N
        self.E=10 #Mev
        self.delta_m2=7.42 #Mev**2
        self.Delta_m2=244 #+-Mev**2
        self.wp=-1/(2*self.E)*self.delta_m2
        self.Wbp=-1/(2*self.E)*self.Delta_m2
        self.k=10**(-17)
        self.mu_R_nu=3.62*10**4#*self.k #Mev
        self.R_mu=32.2#/self.k

        self.starting_point=210.64
        self.nb_shot=10000

        self.occ=3000#1000
        #12000 to reach the same time we did for the classical 4h 45 min
        #3500 to reach t=350, try to see if t=300 is ok too  30 minutes
        #3000 20minutes
        self.dt=0.1#0.1
        self.l_t=[self.dt*i for i in range(self.occ)]

        #self.path="./data_3_flavor_quantum/"
        #self.path="./data_3_flavor_classical/
        self.path="/Users/bauge/Desktop/stage RP/recherche/code/data_3_saveur_Q/"
        self.Path_3saveur="/Users/bauge/Desktop/stage RP/recherche/code/data_3_saveur/"

        self.CIRC=QuantumCircuit(self.NB_qubits,self.NB_qubits)


        self.generate_qi()
        self.generate_qiqi_alpha1_alpha2()
        self.Parameter_PMNS()
        self.compute_PMNS_element()
        self.initialise_state_flavor()

        self.circuit_Initial_state()#init circuit to initial state

        self.circuit_PMNS()
        self.C_Gell_Mann()


    def C_Gell_Mann(self):
        """This function is part of the initialisation process and compute the Gell-Mann matrices.

        Args:
            self (Circuit): Used to store the output:

        Returns:
            Void
            self (Circuit):
                L_Gell_Mann (list of np.matrix): List containing each Gell-Mann matrices in 3*3 dimensions
        """
        l=[]
        l.append(np.matrix([[0,1,0,],[1,0,0],[0,0,0]]))
        l.append(np.matrix([[0,-1j,0],[1j,0,0],[0,0,0]]))
        l.append(np.matrix([[1,0,0],[0,-1,0],[0,0,0]]))
        l.append(np.matrix([[0,0,1],[0,0,0],[1,0,0]]))
        l.append(np.matrix([[0,0,-1j],[0,0,0],[1j,0,0]]))
        l.append(np.matrix([[0,0,0],[0,0,1],[0,1,0]]))
        l.append(np.matrix([[0,0,0],[0,0,-1j],[0,1j,0]]))
        l.append(1/(np.sqrt(3))*np.matrix([[1,0,0],[0,1,0],[0,0,-2]]))
        self.L_Gell_Mann=l
        self.Id3=np.eye(3)

    def Parameter_PMNS(self):
        """This is where we define our mixing angle coefficient needed for the calculation of the PMNS

        Args:
            self(Circuit): Used to store the output

        Return:
            Void
            self(Circuit):
                theta_12 (float): Mixing angle between state 1 and 2 in °
                theta_23 (float): Mixing angle between state 2 and 3 in °
                theta_13 (float): Mixing angle between state 1 and 3 in °
                delta_cp (float): Charge parity violation in °
        """
        def convert_randiant(x):
            """This function convert ° into radian"""
            return 2*np.pi*x/360
        theta12_temp=33.90#33.41
        theta23_temp=48.13#49.1
        theta13_temp=8.52#8.54
        delta_cp_temp=0#197

        self.theta_12=convert_randiant(theta12_temp)
        self.theta_23=convert_randiant(theta23_temp)
        self.theta_13=convert_randiant(theta13_temp)
        self.delta_cp=convert_randiant(delta_cp_temp)

    def compute_PMNS_element(self):
        """This is where we compute the PMNS

        Args:
            self (Circuit): Used to store the output and provide the following input:
                theta_12 (float): Mixing angle between state 1 and 2 in °
                theta_23 (float): Mixing angle between state 2 and 3 in °
                theta_13 (float): Mixing angle between state 1 and 3 in °
                delta_cp (float): Charge parity violation in °

        returns:
            Void
            self (Circuit):
                PMNS_element (np.matrix): 4*4 matrix containing the PMNS matrice corresponding to the encoding used
        """
        c12=np.cos(self.theta_12)
        s12=np.sin(self.theta_12)
        c23=np.cos(self.theta_23)
        s23=np.sin(self.theta_23)
        c13=np.cos(self.theta_13)
        s13=np.sin(self.theta_13)
        ep_delta=np.exp(1j*self.delta_cp)
        em_delta=np.exp(-1j*self.delta_cp)

        self.PMNS_element=np.matrix([[c12*c13,s12*c13,s13*em_delta,0],[-s12*c23-c12*s23*s13*ep_delta,c12*c23-s12*s23*s13*ep_delta,s23*c13,0],[s12*s23-c12*c23*s13*ep_delta,-c12*s23-s12*c23*s13*ep_delta,c23*c13,0],[0,0,0,1]])

    def initialise_state_flavor(self):
        """This is where you set the initial state in the flavor basis

        Args:
            self (Circuit): Used to hold the output and provide the following input:
                N (int): Number of neutrino

        Returns:
            Void
            self (Circuit):
                init_state_flavor (list of int): Initial state of the system, each state is either 0,1 or 2 correspond to each flavor state electron, muon and tau
        """
        if self.N==1:
            State_flavor=[0]
        elif self.N==2:
            State_flavor=[0,1]
        elif self.N==3:
            State_flavor=[0,1,2]
        elif self.N==4:
            State_flavor=[0,0,0,0]
        elif self.N==5:
            State_flavor=[0,0,0,0,0]
        elif self.N==6:
            State_flavor=[0,0,0,0,0,0]
        elif self.N==7:
            State_flavor=[0,0,0,0,0,0,0]
        else:
            print("You don't have an initial state for N=",str(self.N))

        print("the non encoded initial state is:",State_flavor)
        #due to qiskit encoding we also have to reverse the order
        self.init_state_flavor=self.encode_state(State_flavor[::-1])

        print("the encoded state is:", self.init_state_flavor)


        print("The initial state in flavor state is:",self.init_state_flavor)

    def encode_state(self,state_to_encode):
        """This function encode the initial state into the input of the circuit. It's used by the initialise_state_flavor function. The order of the state here use our encoding and the little endian convention.

        Args:
            self (Circuit): No use
            state_to_encode (list of int):  list of element in 0,1 and 2 that correspond to each flavor state

        Return:
            l_temp (list int): List of 0 and 1 corresponding to state 0 or 1 of the qubits we will initialise
            self (Circuit): No use
        """
        l_temp=[]
        for i in range(len(state_to_encode)):
            if state_to_encode[i]==0:
                l_temp.append(0)
                l_temp.append(0)
            elif state_to_encode[i]==1:
                l_temp.append(1)
                l_temp.append(0)
            elif state_to_encode[i]==2:
                l_temp.append(0)
                l_temp.append(1)
            else:
                print("We had an incorrect value:",state_to_encode[i])
        return l_temp

    def compute_2_3_4_qubit_gate(self,s_pauli_gate,l_index,theta):
        """This function create the gate multiple qubits corresponding to Pauli gate

        Args:
            self: Used to get a parameter:
                NB_qubits (int): Number of qubits necessary, corresponding to the number of neutrino
            s_pauli_gate (str): Name of the Pauli gate to compute, "X", "Y" or "Z" for RX,RY or RZ quantum gate
            l_index (list of int): index of all the qubit the gate will work with, each element need to be within NB_qubits and it have the same len as s_pauli_gate, it wont work properly otherwise

        Returns:
            R_circuit (QuantumCircuit): Quantum circuit with two register one quantum and one classical of size NB_qubits. It contain the circuit corresponding to the gate we wanted to compute

        """

        def is_only_I(s_pauli_gate):
            """Boolean check if the gate is only composed of Identity gate"""
            for i in range(len(s_pauli_gate)):
                if s_pauli_gate[i]!="I":
                    return False
            return True

        def filter_I(s_pauli_gate,l_index):
            """This function remove the Identity from the gate.

            Args:
                s_pauli_gate (str): Name of the Pauli gate to compute, "X", "Y" or "Z" for RX,RY or RZ quantum gate
                l_index (list of int): index of all the qubit the gate will work with, each element need to be within NB_qubits and it have the same len as s_pauli_gate, it wont work properly otherwise

            Returns:
                s_out (str): s_pauli_gate with all the "I" removed
                l_index_out (list of int): l_index with the index of all the "I" removed
            """
            #This function remove all the I if their are at least one other pauli gate, else it will make I gate with the corresponding phase
            s_out=""
            l_index_out=[]
            bool_only_I=is_only_I(s_pauli_gate)
            if bool_only_I:
                s_out="I"
                l_index_out=l_index
            else:
                for i in range(len(s_pauli_gate)):
                    if s_pauli_gate[i]!="I":
                        s_out=s_out+s_pauli_gate[i]
                        l_index_out.append(l_index[i])
            return s_out,l_index_out

        l_index=l_index[::-1]#we are converting to little endian notation


        s_pauli_gate,l_index=filter_I(s_pauli_gate,l_index)

        nb_qubit=len(s_pauli_gate)
        R_circuit=QuantumCircuit(self.NB_qubits,self.NB_qubits)
        if nb_qubit==1:
            if s_pauli_gate[0]=="I":#we still have to had the coefficient we have with a phase
                print(l_index)
                R_circuit.rz(theta,l_index[-1])
                R_circuit.p(-theta,l_index[-1])
            if s_pauli_gate[0]=='X':
                R_circuit.rx(theta,l_index[-1])
            if s_pauli_gate[0]=='Y':
                R_circuit.ry(theta,l_index[-1])
            if s_pauli_gate[0]=='Z':
                R_circuit.rz(theta,l_index[-1])

        elif nb_qubit>1:
            for i in range(nb_qubit):
                if s_pauli_gate[i]=='X':#exp(-i theta/2)
                    R_circuit.h(l_index[i])
                elif s_pauli_gate[i]=='Y':
                    R_circuit.p(-np.pi/2,l_index[i])
                    R_circuit.h(l_index[i])
            for i in range(nb_qubit-1):
                if s_pauli_gate[i]!='I':
                    R_circuit.cx(l_index[i],l_index[-1])
            R_circuit.rz(theta,l_index[-1])
            for i in range(1,nb_qubit):
                if s_pauli_gate[i]!='I':
                    R_circuit.cx(l_index[nb_qubit-1-i],l_index[-1])
            for i in range(nb_qubit):
                if s_pauli_gate[i]=='X':
                    R_circuit.h(l_index[i])
                elif s_pauli_gate[i]=='Y':
                    R_circuit.h(l_index[i])
                    R_circuit.p(np.pi/2,l_index[i])
        return R_circuit

    def generate_qi(self):
        """Give the decomposition of the 4*4 matrix of the Qi into Pauli gates

        Args:
            self (Circuit): Used to hold the output

        Returns:
            Void
            self (Circuit):
                l_Qi (list of [list of str],[list of float]): list over all the 8 Qi with the list of their pauli gate name decomposition :Qi[i][0] associated and the list of coefficient associated this pauli gate decomposition Qi[i][1]
        """
        l_Qi=[]
        l_Qi.append([["IX","ZX"],[1/4,1/4]])
        l_Qi.append([["IY","ZY"],[1/4,1/4]])
        l_Qi.append([["IZ","ZZ"],[1/4,1/4]])
        l_Qi.append([["XI","XZ"],[1/4,1/4]])
        l_Qi.append([["YI","YZ"],[1/4,1/4]])
        l_Qi.append([["XX","YY"],[1/4,1/4]])
        l_Qi.append([["YX","XY"],[1/4,-1/4]])
        l_Qi.append([["ZI","ZZ","IZ"],[1/(2*np.sqrt(3)),1/(4*np.sqrt(3)),-1/(4*np.sqrt(3))]])
        self.l_Qi=l_Qi

    def generate_qiqi_alpha1_alpha2(self):
        """This function compute the Qi*Qi terme for the two body interraction. It adopt the same convention as generate_qi for the output

        Args:
            self:
                l_Qi (list of [list of str],[list of float]): list over all the 8 Qi with the list of their pauli gate name decomposition :Qi[i][0] associated and the list of coefficient associated this pauli gate decomposition Qi[i][1]
        Returns:
            Void:
            self (Circuit):
                l_QiQi (list of [list of str],[list of float]): list over all the 8 QiQi with the list of their pauli gate name decomposition :Qi[i][0] associated and the list of coefficient associated this pauli gate decomposition Qi[i][1]
        """
        l_out=[]
        self.generate_qi()
        for i in range(len(self.l_Qi)):
            l_out_gate=[]
            l_out_coeff=[]
            n=len(self.l_Qi[i][0])
            for j in range(n):
                for k in range(n):
                    l_out_gate.append(self.l_Qi[i][0][j]+self.l_Qi[i][0][k])
                    l_out_coeff.append(self.l_Qi[i][1][j]*self.l_Qi[i][1][k])
            l_out.append([l_out_gate,l_out_coeff])

        self.l_QiQi_ab=l_out


    def beta_index(self,i):
        """This function compute give the index of a neutrino return the ordered index of the qubits associated to this neutrino

        Args:
            self (Circuit): No use
            i (int): index in [0,N-1] refering the index of a neutrino

        Returns:
            out (list of integer): list of the two indexis of the qubits associated to the neutrino of index i
        """
        return [2*i,2*i+1]

    def circuit_Initial_state(self):
        """This function compute and add the initialisation circuit to the main circuit CIRC. This function use the initial state define by init_state_flavor and create the initial circuit corresponding to this initial state (0->Id, 1->X gate)

        Args:
            self:
                NB_qubits (int): Number of qubits necessary, corresponding to the number of neutrino
                init_state_flavor (list of int): Initial state of the system, each state is either 0,1 or 2 correspond to each flavor state electron, muon and tau
                CIRC (QuantumCircuit): The main quantum circuit we will use

        Retursn:
            Void
            self (Circuit):
                CIRC QuantumCircuit): We add the initialisation to the main circuit
        """
        for i in range(self.NB_qubits):
            if self.init_state_flavor[i]==1:
                self.CIRC.x(i)
        self.CIRC.barrier()

    def U1_propagator(self,dt):
        """This function compute the U1 circuit corresponding to the one body terme propagator. This function depend greatly on the compute_2_3_4_qubit_gate function.

        Args:
            self:
                NB_qubits (int): Number of qubits necessary, corresponding to the number of neutrino
                N (int): Number of neutrino to consider in the circuit
                l_Qi (list of [list of str],[list of float]): list over all the 8 Qi with the list of their pauli gate name decomposition :Qi[i][0] associated and the list of coefficient associated this pauli gate decomposition Qi[i][1]
                wp (float): Frequency small omega for the one body term in kappa^-1.MeV
                Wbp (float): Frequency big omega for the one body term in kappa^-1.MeV
            dt (float): time step we are using for the propagator

        Returns:
            Void
            self (Circuit):
                U1_step (QuantumCircuit): Circuit corresponding to the U1 propagator propaging of dt, already applied to every neutrino
        """
        #This function construct the circuit corresponding to a specific propagator.
        #It use the l_Qi coefficient and the compute_2_3_4_qubit_gate function to compute the coefficient and the gates we need to use
        c=QuantumCircuit(self.NB_qubits,self.NB_qubits)
        for i in range(self.N):#N-i for the frequency due to qiskit ordering
            for j in range(len(self.l_Qi[2][1])):#these are all two qubits gate
                #print(self.wp)
                #print(dt)
                #print(self.l_Qi[2][1][j])
                coeff3=2*dt*self.wp*(self.N-i)*self.l_Qi[2][1][j]#*25
                #print(coeff3)
                c=c.compose(self.compute_2_3_4_qubit_gate(self.l_Qi[2][0][j],self.beta_index(i),coeff3))#self.beta_index(i)
            for j in range(len(self.l_Qi[7][1])):
                #print(self.Wbp)
                #print(dt)
                #print(self.l_Qi[7][1][j])
                coeff8=2*dt*self.Wbp*(self.N-i)*self.l_Qi[7][1][j]#*np.sqrt(2)#/20
                #print(coeff8)
                c=c.compose(self.compute_2_3_4_qubit_gate(self.l_Qi[7][0][j],self.beta_index(i),coeff8))

        c.barrier()
        #print("U1 propagator is:")
        #print(c)
        #print("U1 propagator done")
        self.U1_step=c

    def function_mu_R_t(self,t):
        """This function compute the coefficient of the two body interraction at the time t shifter by the starting_point"""
        return self.mu_R_nu*(1-np.sqrt(1-(self.R_mu/(t+self.starting_point))**2))**2

    def U2_propagator(self,t):
        """This function compute the circuit corresponding to the propagator U2 taken at a specific time

        Args:
            self:
                NB_qubits (int): Number of qubits necessary, corresponding to the number of neutrino
                N (int): Number of neutrino to consider in the circuit
                dt (float): Size of each step, it needs to be sufficiently small for Trotter to work
                l_QiQi (list of [list of str],[list of float]): list over all the 8 QiQi with the list of their pauli gate name decomposition :Qi[i][0] associated and the list of coefficient associated this pauli gate decomposition Qi[i][1]

        Returns:
            c (QuantumCircuit): Circuit corresponding to the U2 propagator propaging of dt from a time t, already applied to every neutrino
        """
        mu_r_t=self.function_mu_R_t(t+self.dt/2)
        c=QuantumCircuit(self.NB_qubits,self.NB_qubits)
        for a1 in range(self.N):
            for a2 in range(self.N):
                if a1!=a2:
                    l_id=self.beta_index(a1)+self.beta_index(a2)
                    for i in range(8):
                        for j in range(len(self.l_QiQi_ab[i][1])):
                            coeff=1/np.sqrt(2)*2*mu_r_t*self.dt*self.l_QiQi_ab[i][1][j]
                            #coeff=2*mu_r_t*self.dt*self.l_QiQi_ab[i][1][j]
                            c=c.compose(self.compute_2_3_4_qubit_gate(self.l_QiQi_ab[i][0][j],l_id,coeff))
        return c

    def add_measurment(self,U,gate_to_measure,neutrino_to_measure,bool_flavor,do_put_measurement_gate=True):
        """This circuit add the measurement in the appropriate basis to each qubits

        Args:
            self(Circuit):
                NB_qubits (int): Number of qubits necessary, corresponding to the number of neutrino
            U (QuantumCircuit): Circuit we have to add the measurement
            gate_to_measure (str of len 2): Operator we want to measure, by "XY" we assume "III...XY...II" with the placement of XY given by neutrino to measure
            neutrino_to_measure (int): Index of the neutrino to measure
            bool_flavor (boolean): True if we want to convert the measurement back to flavor basis and False otherwise
            do_put_measurement_gate (boolean): True if we want to put the measurement gate, False is needed if we are using statevector

        Returns:
            U_m (QuantumCircuit): The output circuit starting from U and finishing with the measurement
        """
        N=self.NB_qubits
        U_m=U.copy()#we need this!!!
        if bool_flavor:
            #print(self.circuit_PMNS_ajoint())
            U_m=U_m.compose(self.circuit_PMNS_ajoint())

        qubits_to_measure=self.beta_index(neutrino_to_measure)
        for i in range(2):
            if gate_to_measure[i]==0:#X measurement
                U_m.h(qubits_to_measure[i])
            if gate_to_measure[i]==1:#X measurement
                U_m.p(-np.pi/2,qubits_to_measure[i])
                U_m.h(qubits_to_measure[i])
        if do_put_measurement_gate:
            for j in range(N):
                U_m.measure(j,j)
        return U_m

    def add_1step(self,t,do_interraction):
        """Function that add one step of propagation to the main circuit

        Args:
            self (Circuit):
            t (float): time at which we start
            do_interraction (boolean): True if we want to use the interraction term, False otherwise

        Returns:
            Void
            self (Circuit):
                CIRC (QuantumCircuit): Main circuit to which we just add one propagation step
        """
        self.CIRC=self.CIRC.compose(self.step_t(t,do_interraction))

    def step_t(self,t,do_interraction):
        """This function create the circuit corresponding to a sigle step of dt at time t

        Args:
            self (Circuit):
            t (float): time at which we start
            do_interraction (boolean): True if we want to use the interraction term, False otherwise

        Returns:
            circ_out (QuantumCircuit): The circuit corresponding to 1 step of dt at time t of the whole propagator
        """
        c=QuantumCircuit(self.NB_qubits,self.NB_qubits)
        circ_out=c.compose(self.U1_step)
        if do_interraction:
            circ_out=circ_out.compose(self.U2_propagator(t))
        return circ_out

    def circuit_PMNS(self):#flavor to mass
        """This function create the circuit corresponding to the base changes from flavor state to mass state, this circuit actually do the PMNS adjoint.

        Args:
            self (Circuit): Use the mixing angle and the number of neutrino

        Returns:
            Void
            self (Circuit):
                CIRC (QuantumCircuit): Add the flavor to mass basis change to the main circuit.
        """
        c=QuantumCircuit(self.NB_qubits,self.NB_qubits)

        U12=UGate(2*self.theta_12,0,0).control(1)
        U13=UGate(2*self.theta_13,self.delta_cp,-self.delta_cp).control(1)
        U23=UGate(2*self.theta_23,0,0).control(1)
        for i in range(self.N):
            id_temps=self.beta_index(i)
            c.cx(id_temps[1],id_temps[0])
            c.append(U23,[id_temps[0],id_temps[1]])
            c.cx(id_temps[1],id_temps[0])

            c.x(id_temps[0])
            c.append(U13,[id_temps[0],id_temps[1]])
            c.x(id_temps[0])

            c.x(id_temps[1])
            c.append(U12,[id_temps[1],id_temps[0]])
            c.x(id_temps[1])

        c.barrier()
        self.CIRC=self.CIRC.compose(c)

    def circuit_PMNS_ajoint(self):
        """This function create the circuit corresponding to the base changes from mass state to flavor state, this circuit actually do the PMNS.

        Args:
            self (Circuit): Use the mixing angle and the number of neutrino

        Returns:
            Void
            self (Circuit):
                CIRC (QuantumCircuit): Add the mass to flavor basis change to the main circuit.
        """

        c=QuantumCircuit(self.NB_qubits,self.NB_qubits)

        U12=UGate(-2*self.theta_12,0,0).control(1)
        U13=UGate(-2*self.theta_13,self.delta_cp,-self.delta_cp).control(1)
        U23=UGate(-2*self.theta_23,0,0).control(1)
        for i in range(self.N):
            id_temps=self.beta_index(i)
            c.x(id_temps[1])
            c.append(U12,[id_temps[1],id_temps[0]])
            c.x(id_temps[1])

            c.x(id_temps[0])
            c.append(U13,[id_temps[0],id_temps[1]])
            c.x(id_temps[0])

            c.cx(id_temps[1],id_temps[0])
            c.append(U23,[id_temps[0],id_temps[1]])
            c.cx(id_temps[1],id_temps[0])

        c.barrier()
        return c

    def simulate_circuit_all_t(self,gate_to_measure,neutrino_to_measure,bool_flavor=False,do_interraction=False,multi_thread_parameter=0,do_statevector=False):
        """This function do the whole emulation on quantum computer propagation given a set of parameter.

        Args:
            self:
            gate_to_measure (str of len 2): Operator we want to measure, by "XY" we assume "III...XY...II" with the placement of XY given by neutrino to measure
            neutrino_to_measure (int): Index of the neutrino to measure
            bool_flavor (boolean): True if we want to convert the measurement back to flavor basis and False otherwise
            do_interraction (boolean): True if we want to use the interraction term, False otherwise
            multi_thread_parameter (int/boolean): If 1 do the multi_threading process, not working correctly
            do_statevector (boolean): If True use statevector emulation, otherwise use AER emulator

        Returns:
            tab_result (np.array of shape occ * (2**(2N)): Array storing the occurence of each state for all the time. tab_result(t,i) will give the occurence of the state id i at the time t
            l_state (list of (str of len N)): List of the the state corresponding to each index of the tab_result array
        """
        self.CIRC=QuantumCircuit(self.NB_qubits,self.NB_qubits)#initialise the circuit


        self.generate_qi()#neccessary for U1_propagator
        self.Parameter_PMNS()#neccessary for circuit_PMNS
        self.compute_PMNS_element()#neccessary for circuit_PMNS

        self.initialise_state_flavor()

        self.circuit_Initial_state()#add initial state to CIRC
        self.circuit_PMNS()#add PMNS circuit to CIRC
        self.U1_propagator(self.dt)#compute and store self.U1_step we can now use it at will

        #we need a function self.add_measurement() who add the measurement to every qubit

        def generate_binary_ordered_str_permutation(n):#create a dictionnary that given a key of the sort: "1001" return the id for the dimension 1 of the resulte table, it also return the list of all the state ordered
            #n is the number of neutrino
            """This function create a dictionnary that given a state result the id of the state on the tab_result array. It also return the list of theses states.

            Args:
                n (integer): number of qubits

            Returns:
                dicl (dic): The dictionnary that given a state as a key return the corresponding id for this state
                l (list of str): List of all the state ordered by id
            """
            def next_step(last_step):#l whithout encoding, in qutrite encoding
                """Function that compute the permutation list of element in 0,1 of n+1 element given the one at n element"""
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

        #----------------preprocessing
        tab_result=np.zeros((self.occ,2**(2*self.N)))

        key_to_itab_index,l_state=generate_binary_ordered_str_permutation(2*self.N)

        #----------------This section correspond to the case of statevector
        if do_statevector:# we will use state vector instead
            U0=self.CIRC.copy()
            current_state=Statevector(U0)
            measured_state=Statevector(self.add_measurment(U0,gate_to_measure,neutrino_to_measure,bool_flavor,do_put_measurement_gate=False))

            statistics=measured_state.sample_counts(self.nb_shot)
            L_key=list(statistics.keys())
            for j in range(len(L_key)):
                tab_result[0,key_to_itab_index[L_key[j]]]=statistics[L_key[j]]

            tps_debut=time.perf_counter()

            tps1=time.perf_counter()
            for t in range(1,self.occ):
                current_time=self.dt*t
                #process to do for this time step
                U_step=self.step_t(current_time,do_interraction)
                U_t=self.add_measurment(U_step,gate_to_measure,neutrino_to_measure,bool_flavor,do_put_measurement_gate=False)

                #do the process
                measured_state=current_state.evolve(U_t)#the order is important here
                current_state=current_state.evolve(U_step)#step to start from for the next iteration


                #----------------fast process of data
                statistics=measured_state.sample_counts(self.nb_shot)
                L_key=list(statistics.keys())
                for j in range(len(L_key)):
                    tab_result[t,key_to_itab_index[L_key[j]]]=statistics[L_key[j]]

                #----------------Execution time
                if t%50==0 and t!=0:#just to print
                    tps2=time.perf_counter()
                    print("step ",t,"in: ",tps2-tps1,"s")
                    tps1=time.perf_counter()

        #----------------This section correspond to the case where we use the AER simulator
        else:
            #----------------first step
            U_m=self.add_measurment(self.CIRC,gate_to_measure,neutrino_to_measure,bool_flavor)
            #print(U_m)
            result=AerSimulator().run(U_m,shots=self.nb_shot).result()

            #----------------fast process of data
            statistics=result.get_counts()
            L_key=list(statistics.keys())
            for j in range(len(L_key)):
                tab_result[0,key_to_itab_index[L_key[j]]]=statistics[L_key[j]]

            tps_debut=time.perf_counter()

            tps1=time.perf_counter()
            if multi_thread_parameter==0:
                for t in range(1,self.occ):
                    current_time=self.dt*t
                    #----------------one more step
                    self.add_1step(current_time,do_interraction)
                    U_m=self.add_measurment(self.CIRC,gate_to_measure,neutrino_to_measure,bool_flavor)
                    #print(U_m)
                    result=AerSimulator().run(U_m,shots=self.nb_shot).result()

                    #----------------fast process of data
                    statistics=result.get_counts()
                    L_key=list(statistics.keys())
                    for j in range(len(L_key)):
                        tab_result[t,key_to_itab_index[L_key[j]]]=statistics[L_key[j]]

                    #----------------Execution time
                    if t%50==0 and t!=0:#just to print
                        tps2=time.perf_counter()
                        print("step ",t,"in: ",tps2-tps1,"s")
                        tps1=time.perf_counter()
            #----------------This section correspond to the multithreading case
            elif multi_thread_parameter==1:
                nb_core=5
                nb_ram="1GB"
                circuit_packet_size=nb_core*10
                # Initialise your HPC cluster
                my_hpc_cluster = LSFCluster(cores=nb_core, memory=nb_ram)

                # Initialise DASK client with the HPC cluster
                my_dask_client = Client(my_hpc_cluster)

                backend=AerSimulator()
                backend.set_options(executor=my_dask_client,max_job_size=1)
                l_circuit=[]
                l_t=[]
                for t in range(1,self.occ):
                    current_time=self.dt*t
                    #----------------one more step
                    self.add_1step(current_time,do_interraction)
                    U_m=self.add_measurment(self.CIRC,gate_to_measure,neutrino_to_measure,bool_flavor)
                    l_circuit.append(U_m)
                    l_t.append(current_time)
                    current_nb_circuit=len(l_circuit)
                    if (current_nb_circuit>=circuit_packet_size) or (t==self.occ-1):
                        #he we will do the computation for a pack of circuit_packet_size circuit
                        result = backend.run(l_circuit,shots=self.nb_shot).result()


                        #----------------fast process of data
                        out=result.get_counts()
                        for i in range(len(out)):
                            statistics=out[i]
                            L_key=list(statistics.keys())
                            for j in range(len(L_key)):
                                tab_result[t,key_to_itab_index[L_key[j]]]=statistics[L_key[j]]
                        #we reset the packet of circuit
                        l_circuit=[]
                        l_t=[]

                    if t%50==0 and t!=0:#just to print
                        tps2=time.perf_counter()
                        print("step ",t,"in: ",tps2-tps1,"s")
                        tps1=time.perf_counter()
        tps_fin=time.perf_counter()
        print("runing time is:", tps_fin-tps_debut)
        return tab_result,l_state

    #l_state is always the same, it is either 0000,0001,0010 or 0000 1000 0100 depending of the qiskit ordering, it's most likely the second
    def do_all_measurement(self,bool_methode1=False,bool_flavor=False,do_interraction=False,multi_thread_parameter=0,do_statevector=False):
        """This function choose which basis to measure and them simulate the circuit based on this choice. It was part of a deprecated sequence of function. At the end it saved the result table produce by simulat_circuit_all_t in a file.

        Args:
            self (Circuit):
            bool_methode1 (boolean): Check to use another no longer used method
            bool_flavor (boolean): True if we want to convert the measurement back to flavor basis and False otherwise
            do_interraction (boolean): True if we want to use the interraction term, False otherwise
            multi_thread_parameter (int/boolean): If 1 do the multi_threading process, not working correctly
            do_statevector (boolean): If True use statevector emulation, otherwise use AER emulator

        Returns:
            void
            save the tab_result into a file, see save_tab_result


        """
        if bool_methode1:
            print("this method has been removed")
        else:
            print("step "+str(1)+"/1:")
            gate_to_measure,neutrino_to_measure=[2,2],0
            #print(gate_to_measure,neutrino_to_measure)
            tab_result,l_state=self.simulate_circuit_all_t(gate_to_measure, neutrino_to_measure,bool_flavor,do_interraction,multi_thread_parameter,do_statevector)
            #save directly the data afterward, we will compute the operator later
            tab_result=np.c_[self.l_t,tab_result]
            self.save_tab_result(gate_to_measure,neutrino_to_measure,tab_result)

    def save_tab_result(self,gate_to_measure,neutrino_to_measure,tab_result):#gate_neutrino_nb_neutrino
        """This function save the array tab_result into a .npy file. The file is identify by the gate we measured, the neutrino we measured specifically and the number of neutrino used in the calculation

        Args:
            self (Circuit):
            gate_to_measure (str of len 2): Operator we want to measure, by "XY" we assume "III...XY...II" with the placement of XY given by neutrino to measure
            neutrino_to_measure (int): Index of the neutrino to measure
            tab_result (np.array of shape occ * (2**(2N)): Array storing the occurence of each state for all the time. tab_result(t,i) will give the occurence of the state id i at the time t

        Returns:
            void
            save the tab_result into a .npy file
        """
        name_file="phi_gate_"
        for i in range(2):
            if gate_to_measure[i]==0:#X measurement
                name_file=name_file+"X"
            if gate_to_measure[i]==1:#X measurement
                name_file=name_file+"Y"
            if gate_to_measure[i]==2:#X measurement
                name_file=name_file+"Z"
        name_file=name_file+"_"+str(neutrino_to_measure)+"_"
        with open(self.path+name_file+str(self.N)+".npy", 'wb') as f:
            np.save(f, tab_result)

    def load_tab_result(self,gate_to_measure,neutrino_to_measure):
        """This function load back the tab_result array.

        Args:
            self (Circuit):
            gate_to_measure (str of len 2): Operator we want to measure, by "XY" we assume "III...XY...II" with the placement of XY given by neutrino to measure
            neutrino_to_measure (int): Index of the neutrino to measure
        Returns:
            tab_result (np.array of shape occ * (2**(2N)): Array storing the occurence of each state for all the time. tab_result(t,i) will give the occurence of the state id i at the time t
        """
        name_file="phi_gate_"
        for i in range(2):
            if gate_to_measure[i]==0:#X measurement
                name_file=name_file+"X"
            if gate_to_measure[i]==1:#X measurement
                name_file=name_file+"Y"
            if gate_to_measure[i]==2:#X measurement
                name_file=name_file+"Z"
        name_file=name_file+"_"+str(neutrino_to_measure)+"_"

        with open(self.path+name_file+str(self.N)+".npy", 'rb') as f:
            tab_result=np.load(f)
        return tab_result

    def measure_operator(self):
        """This function measure the state 00,01 and 10 in the Z basis corresponding to each state of your encoding.
        Args:
            self (Circuit):
            load:
                tab_1=tab_result (np.array of shape occ * (2**(2N)): Array storing the occurence of each state for all the time. tab_result(t,i) will give the occurence of the state id i at the time t

        Returns:
            tab_out (array of size 3*N*occ): tab storing the value of each state measurement i, for each neutrino n, for each time t.
            tab_out[i,n,t] give the value of the state i for the neutrino number n at time t.
            Due to the basis in which we do the measurement it will either be the mass or the flavor state
        """
        l_tab=[]
        #we need to know the state order
        def generate_binary_ordered_str_permutation(n):
            """This function create a dictionnary that given a state result the id of the state on the tab_result array. It also return the list of these state.

            Args:
                n (integer): number of qubits

            Returns:
                dicl (dic): The dictionnary that given a state as a key return the corresponding id for this state
                l (list of str): List of all the state ordered by id
            """
            def next_step(last_step):#l whithout encoding, in qutrite encoding
                """Function that compute the permutation list of element in 0,1 of n+1 element given the one at n element"""
                l_out=[]
                for i in last_step:
                    l_out.append('0'+i)
                    l_out.append('1'+i)
                return l_out
            l=['']
            for i in range(n):
                l=next_step(l)
            l=np.sort(l)
            return l
        def l_string_reverse(l):
            """This function reverse the element order in a list"""
            for i in range(len(l)):
                l[i]=l[i][::-1]
            return l
        l_state=generate_binary_ordered_str_permutation(2*self.N)
        print(l_state)
        #l_state=l_string_reverse(l_state)
        tab_out=np.zeros((3,self.N,self.occ))
        tab_1=self.load_tab_result([2,2],0)
        for n in range(self.N):#for all neutrino
            print(tab_1)
            for t in range(self.occ):#for all time
                id12=self.beta_index(n)
                id1=id12[0]
                id2=id12[1]
                f1=0
                f2=0
                f3=0
                for i in range(len(l_state)):#for all the state
                    if l_state[i][id1]+l_state[i][id2]=="00":
                        f1=f1+tab_1[t][i+1]#+1 because first collumns is time
                    if l_state[i][id1]+l_state[i][id2]=="01":
                        f2=f2+tab_1[t][i+1]#+1 because first collumns is time
                    if l_state[i][id1]+l_state[i][id2]=="10":
                        f3=f3+tab_1[t][i+1]#+1 because first collumns is time
                tab_out[0,n,t]=f1/self.nb_shot
                tab_out[1,n,t]=f2/self.nb_shot
                tab_out[2,n,t]=f3/self.nb_shot
        return(tab_out)

    def save_tab_mass_or_flavor_state(self,tab_in,is_flavor,is_interraction):
        """This function save the measurement of the state, either in mass or flavor state in a file where the name containt the basis of measurement, the presence or not of interraction, the number of neutrino the value of dt and the number of dt step"""
        if is_flavor:
            name_file="tab_flavor"
        else:
            name_file="tab_mass"
        name_file=name_file+"_state_"+"occ="+str(self.occ)+"_dt="+str(self.dt)
        if is_interraction:
            name_file=name_file+"_inter_N="+str(self.N)+".npy"
        else:
            name_file=name_file+"_N="+str(self.N)+".npy"
        with open(self.path+name_file, 'wb') as f:
            np.save(f, tab_in)

    def load_tab_mass_or_flavor_state(self,is_flavor,is_interraction):
        """This function will load the result of the measurement of the state based on the parameter of the circuit associated to self"""
        if is_flavor:
            name_file="tab_flavor"
        else:
            name_file="tab_mass"
        name_file=name_file+"_state_"+"occ="+str(self.occ)+"_dt="+str(self.dt)
        if is_interraction:
            name_file=name_file+"_inter_N="+str(self.N)+".npy"
        else:
            name_file=name_file+"_N="+str(self.N)+".npy"
        with open(self.path+name_file, 'rb') as f:
            tab_out=np.load(f)
        return tab_out


    #---------------------------------Ploting function

    def plot_result(self,tab,n_neutrino):
        """This function take the output of the measure_operator as tab and plot the value of each state for the neutrino n_neutrino

        Args:
            tab (array of size 3*N*occ): tab storing the value of each state measurement i, for each neutrino n, for each time t.
            tab[i,n,t] give the value of the state i for the neutrino number n at time t.
            Due to the basis in which we do the measurement it will either be the mass or the flavor state
            n_neutrino (int): index of the neutrino we want to plot the state of

        Returns:
            Void
            plot a graph
        """
        l_t=[i*self.dt+self.starting_point for i in range(self.occ)]
        plt.plot(l_t,tab[0,n_neutrino],label="0")
        plt.plot(l_t,tab[1,n_neutrino],label="1")
        plt.plot(l_t,tab[2,n_neutrino],label="2")
        plt.ylim([0,1])
        plt.legend()
        plt.show()
        return(0)

    def multiplot_pnu_state(self,tab_in,bool_flavor=True):
        """This function plot three graph in a vertical figure where on each graph we plot the evolution of each of the three state (either mass or flavor) for all the neutrino.

        Args:
            self (Circuit):
            tab_in (array of size 3*N*occ): tab storing the value of each state measurement i, for each neutrino n, for each time t.
            tab_in[i,n,t] give the value of the state i for the neutrino number n at time t.
            Due to the basis in which we do the measurement it will either be the mass or the flavor state
            bool_flavor (boolean): tell which of the flavor or the mass basis we are working with

        Returns:
            void
            plot the graph
        """
        l_t=[i*self.dt+self.starting_point for i in range(self.occ)]
        tab_P_e=tab_in[0]
        tab_P_mu=tab_in[1]
        tab_P_tau=tab_in[2]

        l_color=list(mcolors.TABLEAU_COLORS.values())
        fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
        for i in range(self.N):
            axs[0].plot(l_t,tab_P_e[i,:],l_color[i],linestyle="-",label="neutrino "+str(i))

            axs[1].plot(l_t,tab_P_mu[i,:],l_color[i],linestyle="-",label="neutrino "+str(i))

            axs[2].plot(l_t,tab_P_tau[i,:],l_color[i],linestyle="-",label="neutrino "+str(i))

        axs[-1].set_xlabel("t(kappa ^{-1})", fontsize=18)
        if bool_flavor!=True:
            axs[0].set_ylabel("$P_{e}$")
            axs[1].set_ylabel("P_{mu}")
            axs[2].set_ylabel("$P_{\tau}$")
        else:
            axs[0].set_ylabel("$P_{1}$")
            axs[1].set_ylabel("$P_{2}$")
            axs[2].set_ylabel("$P_{3}$")

        axs[0].legend()
        axs[1].legend()
        axs[2].legend()

        plt.show()

    def multiplot_pnu_state_mass_state_comparaison(self,tab_in):
        """This function plot three graph in a vertical figure where on each graph we plot the evolution of each of the three mass state for all the neutrino. It also load the classical result superposed them with the quantum result. We went further in time for the classical result, so we will discard any point that go further than the quantum result to increase visibility.

        Args:
            self (Circuit):
            tab_in (array of size 3*N*occ): tab storing the value of each state measurement i, for each neutrino n, for each time t.
            tab_in[i,n,t] give the value of the state i for the neutrino number n at time t.
            Due to the basis in which we do the measurement it will either be the mass or the flavor state
            bool_flavor (boolean): tell which of the flavor or the mass basis we are working with

        Returns:
            void
            plot the graph
        """

        tab_phi,tab_P_nu1,tab_P_nu2,tab_P_nu3=self.load_classical_result_mass_state()

        l_t=[i*self.dt for i in range(self.occ)]
        tab_P_e=tab_in[0]
        tab_P_mu=tab_in[1]
        tab_P_tau=tab_in[2]

        l_t2=tab_phi[:,0]
        max_t=self.occ*self.dt
        print(max_t)
        def seek_id_max(l,max_element):
            """This function return the maximal index we need to consider on the classical result"""
            for i in range(len(l)):
                if l[i]>max_element:
                    return i
            return len(l)
        max_id=seek_id_max(l_t2,max_t)

        l_color=list(mcolors.TABLEAU_COLORS.values())
        fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
        for i in range(self.N):
            axs[0].plot(l_t,tab_P_e[i,:],l_color[i],linestyle="-",label="neutrino "+str(i)+"quantum")
            axs[0].plot(l_t2[:max_id],tab_P_nu1[:max_id,i],l_color[i+self.N],label="neutrino "+str(i)+"classical",linestyle="-")

            axs[1].plot(l_t,tab_P_mu[i,:],l_color[i],linestyle="-",label="neutrino "+str(i)+"quantum")
            axs[1].plot(l_t2[:max_id],tab_P_nu2[:max_id,i],l_color[i+self.N],label="neutrino "+str(i)+"classical",linestyle="-")

            axs[2].plot(l_t,tab_P_tau[i,:],l_color[i],linestyle="-",label="neutrino "+str(i)+"quantum")
            axs[2].plot(l_t2[:max_id],tab_P_nu3[:max_id,i],l_color[i+self.N],label="neutrino "+str(i)+"classical",linestyle="-")

        axs[-1].set_xlabel("$t(\kappa ^{-1}$)", fontsize=18)

        axs[0].set_ylabel("$P_{1}$",fontsize=18)
        axs[1].set_ylabel("$P_{2}$",fontsize=18)
        axs[2].set_ylabel("$P_{3}$",fontsize=18)
        """
        axs[0].set_ylim([0,1])
        axs[1].set_ylim([0,1])
        axs[2].set_ylim([0,1])
        """
        #axs[0].legend()
        #axs[1].legend()
        axs[2].legend()

        plt.show()

    def load_classical_result(self):
        """This function load the classical result of the flavor state based on the value of N.

        Args:
            self (Circuit): Only used for the value of N

        Returns:
            out (array of shape (nb_t,N)): This array contain the calculated classical wavefunction, it's will only be used to extract the list of all the time
            tab_P_e (array of shape (nb_t,N)): Containt the population of the first flavor state. tab_P_e[t,i] will give the value of the first flavor state at time t for the neutrino i.
            tab_P_mu (array of shape (nb_t,N)): Containt the population of the second flavor state. tab_P_mu[t,i] will give the value of the second flavor state at time t for the neutrino i.
            tab_P_tau (array of shape (nb_t,N)): Containt the population of the third flavor state. tab_P_tau[t,i] will give the value of the third flavor state at time t for the neutrino i.
        """
        print("loading phi_t_flavor")
        with open(self.Path_3saveur+"t_phi_t_flavor="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            out=np.load(f)
        with open(self.Path_3saveur+"tab_P_e="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            tab_P_e=np.load(f)
        with open(self.Path_3saveur+"tab_P_mu="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            tab_P_mu=np.load(f)
        with open(self.Path_3saveur+"tab_P_tau="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            tab_P_tau=np.load(f)
        return out,tab_P_e,tab_P_mu,tab_P_tau

    def load_classical_result_mass_state(self):
        """This function load the classical result of the mass state based on the value of N.

        Args:
            self (Circuit): Only used for the value of N

        Returns:
            out (array of shape (nb_t,N)): This array contain the calculated classical wavefunction, it's will only be used to extract the list of all the time
            tab_P_e (array of shape (nb_t,N)): Containt the population of the first mass state. tab_P_e[t,i] will give the value of the first mass state at time t for the neutrino i.
            tab_P_mu (array of shape (nb_t,N)): Containt the population of the second mass state. tab_P_mu[t,i] will give the value of the second mass state at time t for the neutrino i.
            tab_P_tau (array of shape (nb_t,N)): Containt the population of the third mass state. tab_P_tau[t,i] will give the value of the third mass state at time t for the neutrino i.
        """
        print("loading phi_t_mass")
        with open(self.Path_3saveur+"t_phi_t_flavor="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            out=np.load(f)
        with open(self.Path_3saveur+"tab_P_nu1="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            tab_P_e=np.load(f)
        with open(self.Path_3saveur+"tab_P_nu2="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            tab_P_mu=np.load(f)
        with open(self.Path_3saveur+"tab_P_nu3="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            tab_P_tau=np.load(f)
        return out,tab_P_e,tab_P_mu,tab_P_tau

    def plot_proba_state(self,tab_P_e,tab_P_mu,tab_P_tau,tab_phi_t_flavor,tab):
        """This function will plot the evolution in time of the population of each flavor state for a single neutrino. It can be used to plot neutrino oscillation for a single neutrino when we are using data without interraction. This function will compare it with the classical computation.

        Args:
            self (Circuit):
            tab_P_e (array of shape (nb_t,N)): Containt the population of the first flavor state. tab_P_e[t,i] will give the value of the first flavor state at time t for the neutrino i.
            tab_P_mu (array of shape (nb_t,N)): Containt the population of the second flavor state. tab_P_mu[t,i] will give the value of the second flavor state at time t for the neutrino i.
            tab_P_tau (array of shape (nb_t,N)): Containt the population of the third flavor state. tab_P_tau[t,i] will give the value of the third flavor state at time t for the neutrino i.
            tab_phi_t_flavor (array of shape (nb_t,N)): This array contain the calculated classical wavefunction, it's will only be used to extract the list of all the time
            tab (array of size 3*N*occ): tab storing the value of each state measurement i, for each neutrino n, for each time t.
            tab_in[i,n,t] give the value of the state i for the neutrino number n at time t.

        Returns:
            Void
            plot the graph comparing the flavor oscillation between the classical and the quantum result
        """

        l_t=tab_phi_t_flavor[:,0]
        max_t=self.occ*self.dt
        def seek_id_max(l,max_element):
            for i in range(len(l)):
                if l[i]>max_element:
                    return i
            return len(l)
        max_id=seek_id_max(l_t,max_t)
        l_color=list(mcolors.TABLEAU_COLORS.values())
        l_style=["-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":"]
        plt.plot(l_t[:max_id],tab_P_e[:max_id,0],l_color[0],label="$n_{e}$ classical",linestyle="None",marker=".",markersize=5)
        plt.plot(l_t[:max_id],tab_P_mu[:max_id,0],l_color[1],label="$n_{mu}$ classical",linestyle="None",marker=".",markersize=5)
        plt.plot(l_t[:max_id],tab_P_tau[:max_id,0],l_color[2],label="$n_{tau}$ clasical",linestyle="None",marker=".",markersize=5)

        l_t2=[i*self.dt for i in range(self.occ)]
        plt.plot(l_t2,tab[0,0],l_color[0],label="$n_{e}$ quantum",linestyle="-")
        plt.plot(l_t2,tab[1,0],l_color[1],label="$n_{mu}$ quantum",linestyle="-")
        plt.plot(l_t2,tab[2,0],l_color[2],label="$n_{tau}$ quantum",linestyle="-")


        plt.xlabel("kappa^{-1}",fontsize=18)
        plt.ylabel("Probability",fontsize=18)
        plt.legend()
        #plt.xscale("log")
        plt.show()
##test
"""This section concern some function used to test the earlier function and do some calculation by hand faster
"""

Sx=np.matrix([[0,1],[1,0]])
Sy=np.matrix([[0,-1j],[ 1j,0]])
Sz=np.matrix([[1,0],[0,-1]])
Ha=1/np.sqrt(2)*np.matrix([[1,1],[1,-1]])
Id2=np.eye(2)

def c(s): #string of the form XIYZ return the matrice corresponding to it
    """This function compute the matrix associated the string s given.

    Args:
        s (str): character can take value in "X","Y","Z","I" and "H" describing the order of operation for the kronecker product

    Returns:
        m (matrix): Matrix associated to the string of character

    """
    m=np.matrix([[1]])
    for i in s:
        if i=="X":
            m=np.kron(m,Sx)
        if i=="Y":
            m=np.kron(m,Sy)
        if i=="Z":
            m=np.kron(m,Sz)
        if i=="I":
            m=np.kron(m,Id2)
        if i=="H":
            m=np.kron(m,Ha)
    return m

def c2(s):
    """This function was used to check the convention of qiskit and more specificaly build the circuit corresponding to the PMNS"""
    n=4
    m=np.eye(4)
    coeff=-0.5
    for i in s:
        if i=="c":#small c for cx on 1
            qc=QuantumCircuit(2)
            qc.cx(0,1)
            c=qi.Operator(qc)
            m=np.matmul(m,c)
        if i=="C":#big C for cx on 1
            qc=QuantumCircuit(2)
            qc.cx(1,0)
            C=qi.Operator(qc)
            #print(qc)
            m=np.matmul(m,C)
        if i=="U":
            G=UGate(2*coeff,0,0).control(1)
            qc=QuantumCircuit(2)
            #qc.x(0)
            qc.append(G,[0,1])
            #qc.x(0)
            U=qi.Operator(qc)
            #print(qc)
            #print(U)
            #U=np.matrix([[np.cos(coeff),-np.sin(coeff),0,0],[np.sin(coeff),np.cos(coeff),0,0],[0,0,1,0],[0,0,0,1]])
            m=np.matmul(m,U)
        if i=="X":
            qc=QuantumCircuit(2)
            qc.x(0)
            a=qi.Operator(qc)
            m=np.matmul(m,a)
        if i=="x":
            qc=QuantumCircuit(2)
            qc.x(1)
            a=qi.Operator(qc)
            m=np.matmul(m,a)
        if i=="u":
            G=UGate(2*coeff,0,0).control(1)
            qc=QuantumCircuit(2)
            #qc.x(1)
            qc.append(G,[1,0])

            #qc.x(1)
            u=qi.Operator(qc)
            #print(qc)
            #print(U)
            #U=np.matrix([[np.cos(coeff),-np.sin(coeff),0,0],[np.sin(coeff),np.cos(coeff),0,0],[0,0,1,0],[0,0,0,1]])

            #u=np.matrix([[np.cos(coeff),-np.sin(coeff),0,0],[0,1,0,0],[np.sin(coeff),0,np.cos(coeff),0],[0,0,0,1]])
            m=np.matmul(m,u)
        if i=="S":
            qc=QuantumCircuit(2)
            qc.swap(1,0)
            S=qi.Operator(qc)
            #S=np.matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
            m=np.matmul(m,S)
    #print(m)
    return m
def p(s1,s2):
    """Do the product between the matrix s1 and s2."""
    return np.matmul(s1,s2)

def time_estimation(nb_50_occ):
    """This function give a rought time estimation for the non state vector case given a the time to do the first 50 step of calculation b the difference between the time to do the 50 to 100 step and the time to do 50 step.

    Args:
        nb_50_occ (int): is the number of time we will do 50 step of calculation. For example for occ=5000, we will do nb_50_occ=100 step.
    """
    a=25
    b=45
    print("pour:",nb_50_occ*50," steps")
    l=[a+i*b for i in range(nb_50_occ)]
    return sum(l)



##Application
"""This section contain some already prepared code to directly use the function presented ealier.

Before executing the code, you need to check the path at which data will save and load from.
    - The "path" correspond to the result of this code
    - The "Path_3saveur" correspond to the result of the classical code

In order to use them you can interract with the following parameter:
-initial state written in the initialise_state_flavor method
-mixing angle written in the Parameter_PMNS method
-number of neutrino by changing the value of n_neutrino below or when calling Circuit(n)
-calculation methode:
    * do_statevector1 : True if we want to use statevector from qiskit
    * do_interraction_1 : True if we want to use the interraction term
    * bool_flavor_1 : True if we want to compute the circuit when the ouput is in flavor state, mass state otherwise
-Rest of the coefficient link to the simulation: __init__
    * occ :number of time step to do
    * dt : time step to do
    * parameter of the U1 and U2 propagator
If we want to modify hamiltonian:
-You can modify the one body term in the methode U1_propagator as long as it's not time dependent it should be correct
-You can modify the interraction term in U2_propagator (Here the limitation of this implementation is that all coefficient of the matrice have the same time dependance. H2=f(t)*matrix with f(t) a function that return a scalar. In our case f(t) is function_mu_R_t)
-When you run the code, it save and load the quantum result from the file at the path "path" defined in the _init_ method of the Circuit class.
-For methode 9, some graph necessitate to load the corresponding classical result, to do that you need to put the classical result at the path "Path_3saveur" defined in the _init_ method of the Circuit class.


The mode parameter control which work you want to do:

* mode==5: This code will do the whole simulation process corresponding to the parameter define ealier and save the result. It doesn't compute the measurement. Result will be saved at path.

* mode==6: This code will load the result of the simulation and compute the measurement of each state given the parameter define ealier. It doesn't compute the simulation, so it will not work if you don't already run mode==5 with cohenrent parameter. Result will be load from path and saved at path

*mode==7: This code will do both the simulation and the measurement of each state. At the end it save the state. Result will saved at path.

*mode==9: This code is task to plot all the graph, the last plot may not work if you didn't already compute the related classical result. Quantum result will be load from path, classical result will be load from Path_3saveur.


In order to observe the flavor oscillation without interraction, it's advised to run the code with a single neutrino with do_interraction as False and bool_flavor_1=True (methode 7 then 9).
In order to observe the effect of the interraction term, the result of the interraction will be visible on the mass state, so it's advised to run the code with bool_flavor=False.
"""

mode=9

n_neutrino=1

#ignore these parameters, they are not used
bool_methode1_1=False#never pass to True, it doesn't work for now
multi_thread_parameter_1=0#0 is no multi_threadinf,1 is do multithreding
#working but with some issues, take more time than a single thread in our case


do_statevector1=True

do_interraction_1=True
bool_flavor_1=True

circ=Circuit(n_neutrino)
#------------------methode 2 work for mass basis
if mode==5: #work methode 2 with same measurement
    circ.do_all_measurement(bool_methode1=bool_methode1_1,bool_flavor=bool_flavor_1,do_interraction=do_interraction_1,multi_thread_parameter=multi_thread_parameter_1,do_statevector=do_statevector1)

    tab_out=circ.measure_operator()
elif mode==6: #methode 2  without measurement, only need the ZZ measurement
    tab_out=circ.measure_operator()

elif mode==7:#compute everything and save
    circ.do_all_measurement(bool_methode1=bool_methode1_1,bool_flavor=bool_flavor_1,do_interraction=do_interraction_1,multi_thread_parameter=multi_thread_parameter_1,do_statevector=do_statevector1)
    tab_out=circ.measure_operator()
    circ.save_tab_mass_or_flavor_state(tab_out,bool_flavor_1,do_interraction_1)

elif mode==9:#load and plot
    tab_out=circ.load_tab_mass_or_flavor_state(bool_flavor_1,do_interraction_1)

    if bool_flavor_1==False:
        circ.multiplot_pnu_state_mass_state_comparaison(tab_out)
    circ.multiplot_pnu_state(tab_out,bool_flavor=bool_flavor_1)

    circ.plot_result(tab_out,0)
    #comparaison exact
    tab_phi_t,tab_P_e,tab_P_mu,tab_P_tau=circ.load_classical_result()
    circ.plot_proba_state(tab_P_e,tab_P_mu,tab_P_tau,tab_phi_t,tab_out)