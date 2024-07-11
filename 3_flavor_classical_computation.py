"""This code is tasked to compute classicaly (in opposition to quantum emulation) the three flavor oscillation of neutrino with the two body interraction.
This code regroupe every function and methode related to the classical computation of the Hamiltonian introduced by Siwach and al in the following publication: (https://journals.aps.org/prd/abstract/10.1103/PhysRevD.107.023019).
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
"""

##Include
import math
import numpy as np
import time
from scipy.linalg import expm,logm
#from qiskit.quantum_info import partial_trace
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt

##partial trace
"""In this section we introduced a function to computer the n-partite partial trace that come from an old version of qiskit, the current implementation of partial trace on qiskit is only able to do bi-partite matrice. Here, as we are working with qutrite, it's more convenient to work with them.
As i copied the next three function from the qiskit repository of the 0.14 stable version, here is the link to this repositery:
https://github.com/Qiskit/qiskit/blob/stable/0.14/qiskit/tools/qi/qi.py
And here is the copyright associated with the code.
"""
# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name,unsupported-assignment-operation
"""
A collection of DEPRECATED quantum information functions.

Please see the `qiskit.quantum_info` module for replacements.
"""

def partial_trace(state, trace_systems, dimensions=None, reverse=True):
    """
    Partial trace over subsystems of multi-partite matrix.

    Note that subsystems are ordered as rho012 = rho0(x)rho1(x)rho2.

    Args:
        state (matrix_like): a matrix NxN
        trace_systems (list(int)): a list of subsystems (starting from 0) to
                                  trace over.
        dimensions (list(int)): a list of the dimensions of the subsystems.
                                If this is not set it will assume all
                                subsystems are qubits.
        reverse (boolean): ordering of systems in operator.
            If True system-0 is the right most system in tensor product.
            If False system-0 is the left most system in tensor product.

    Returns:
        matrix_like: A density matrix with the appropriate subsystems traced
            over.
    Raises:
        Exception: if input is not a multi-qubit state.
    """

    if dimensions is None:  # compute dims if not specified
        num_qubits = int(np.log2(len(state)))
        dimensions = [2 for _ in range(num_qubits)]
        if len(state) != 2**num_qubits:
            raise Exception("Input is not a multi-qubit state, "
                            "specify input state dims")
    else:
        dimensions = list(dimensions)

    if isinstance(trace_systems, int):
        trace_systems = [trace_systems]
    else:  # reverse sort trace sys
        trace_systems = sorted(trace_systems, reverse=True)

    # trace out subsystems
    if state.ndim == 1:
        # optimized partial trace for input state vector
        return __partial_trace_vec(state, trace_systems, dimensions, reverse)
    # standard partial trace for input density matrix
    return __partial_trace_mat(state, trace_systems, dimensions, reverse)


def __partial_trace_vec(vec, trace_systems, dimensions, reverse=True):
    """
    Partial trace over subsystems of multi-partite vector.

    Args:
        vec (vector_like): complex vector N
        trace_systems (list(int)): a list of subsystems (starting from 0) to
                                  trace over.
        dimensions (list(int)): a list of the dimensions of the subsystems.
                                If this is not set it will assume all
                                subsystems are qubits.
        reverse (bool): ordering of systems in operator.
            If True system-0 is the right most system in tensor product.
            If False system-0 is the left most system in tensor product.

    Returns:
        ndarray: A density matrix with the appropriate subsystems traced over.
    """

    # trace sys positions
    if reverse:
        dimensions = dimensions[::-1]
        trace_systems = len(dimensions) - 1 - np.array(trace_systems)

    rho = vec.reshape(dimensions)
    rho = np.tensordot(rho, rho.conj(), axes=(trace_systems, trace_systems))
    d = int(np.sqrt(np.product(rho.shape)))

    return rho.reshape(d, d)


def __partial_trace_mat(mat, trace_systems, dimensions, reverse=True):
    """
    Partial trace over subsystems of multi-partite matrix.

    Note that subsystems are ordered as rho012 = rho0(x)rho1(x)rho2.

    Args:
        mat (matrix_like): a matrix NxN.
        trace_systems (list(int)): a list of subsystems (starting from 0) to
                                  trace over.
        dimensions (list(int)): a list of the dimensions of the subsystems.
                                If this is not set it will assume all
                                subsystems are qubits.
        reverse (bool): ordering of systems in operator.
            If True system-0 is the right most system in tensor product.
            If False system-0 is the left most system in tensor product.

    Returns:
        ndarray: A density matrix with the appropriate subsystems traced over.
    """

    trace_systems = sorted(trace_systems, reverse=True)
    for j in trace_systems:
        # Partition subsystem dimensions
        dimension_trace = int(dimensions[j])  # traced out system
        if reverse:
            left_dimensions = dimensions[j + 1:]
            right_dimensions = dimensions[:j]
            dimensions = right_dimensions + left_dimensions
        else:
            left_dimensions = dimensions[:j]
            right_dimensions = dimensions[j + 1:]
            dimensions = left_dimensions + right_dimensions
        # Contract remaining dimensions
        dimension_left = int(np.prod(left_dimensions))
        dimension_right = int(np.prod(right_dimensions))

        # Reshape input array into tri-partite system with system to be
        # traced as the middle index
        mat = mat.reshape([
            dimension_left, dimension_trace, dimension_right, dimension_left,
            dimension_trace, dimension_right
        ])
        # trace out the middle system and reshape back to a matrix
        mat = mat.trace(axis1=1,
                        axis2=4).reshape(dimension_left * dimension_right,
                                         dimension_left * dimension_right)
    return mat

##class definition
#this code is in SU(3)


class Hamiltionian_classique_SU3:
    """This class is tasked to store one Hamiltonian and introduce some methode to compute it's propagation other time.
    The calculation is not done in the same encoding as the quantum computation. Here the encoding used is an SU(3) encoding, like using qutrite.
    """
    def __init__(self,n):
        """This initialisation function will define some parameter and intialise some calculation.

        Args:
            n (int): number of neutrino to consider in the circuit.

        Variables:
            N (int): number of neutrino to consider in the circuit
            E (float): Energy of each single neutrino we consider in MeV
            delta_m2 (float): Mass difference between the mass state 2 and 1 in kappa^-1. MeV**2
            Delta_m2 (float): Mass difference between the mass state 3 and the mass state 2 in kappa^-1.MeV**2
            wp (float): Frequency small omega for the one body term in kappa^-1.MeV
            Wbp (float): Frequency big omega for the one body term in kappa^-1.MeV
            k (float): Arbitrary unit introduce by siwach, correspond to 19.7 km/MeV
            mu_R_nu (float): main coefficient of the interraction terme
            R_mu (float): Secondary coefficient of the interraction term
            starting_point (float): Shift in the interraction to start from, we can't start from 0 because the interraction would diverge to infinity
            t_start (float): time from where to start
            t_finish (float): time at which to end
            dt (float): Size of each step, it needs to be sufficiently small for Trotter and RK4 to work
            t_nb_step (int): computed from t_finish,t_start and dt. Is the number of step needed.
            Path_3saveur (str): Path to load the data of classical computation from
        """
        self.N=n #nb neutrino #for more than 7 neutrino, some calculation are slow down by the memory of the camputer (8go of ram only)
        self.E=10 #Mev
        self.delta_m2=7.42#*10**(-17) #Mev**2
        self.Delta_m2=244#*10**(-17) #+-Mev**2
        self.wp=-1/(2*self.E)*self.delta_m2
        self.Wbp=-1/(2*self.E)*self.Delta_m2
        self.k=10**(-17)
        self.mu_R_nu=3.62*10**4#*self.k #Mev
        self.R_mu=32.2#/self.k

        self.starting_point_methode1=210.64#/self.k

        self.t_start=0
        self.t_finish=1000#1000

        self.dt=0.001#0.001#0.001
        self.t_nb_step=int(np.ceil((self.t_finish-self.t_start)/self.dt))

        #self.path="./data_3_flavor_classical/
        self.Path_3saveur="/Users/bauge/Desktop/stage RP/recherche/code/data_3_saveur/"

        #initialise some constant of the parameters
        self.Time_interval()
        self.Pauli_matrices()
        self.C_Gell_Mann()
        self.Compute_all_Q_i()
        self.C_PMNS()

        #prepare the initial state
        self.init_Phi0()

        #compute H_nu and H_nunu
        self.compute_H_element()

    def Time_interval(self):
        """This function compute the list of time given the time related parameter.

        Args:
            self:
                t_start (float): time from where to start
                t_finish (float): time at which to end
                t_nb_step (int): computed from t_finish,t_start and dt. Is the number of step needed.

        Returns:
            Void
            self:
                l_t (list of float): list of all the point we will propagate through
        """
        self.l_t=np.linspace(self.t_start,self.t_finish,self.t_nb_step)


    def Pauli_matrices(self):
        """This function define Sx,Sy,Sz and Id2 which are the pauli matrices and 2*2 identity"""
        self.Sx=np.matrix([[0,1],[1,0]])
        self.Sy=np.matrix([[0,-1j],[ 1j,0]])
        self.Sz=np.matrix([[1,0],[0,-1]])
        self.Id2=np.eye(2)

    def C_Gell_Mann(self):
        """This function define the Gell-Mann matrices in a list.

        Returns:
            void
            self:
                L_Gell_Mann (list of matrix (3*3)): List of the 8 gell mann matrices.
                Id3 (matrix): identity 3*3
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

    def Compute_all_Q_i(self):
        """This function compute the Q_i matrices from the Gell-Mann matrices applied to a single neutrino.

        Args:
            self:
                L_Gell_Mann (list of matrix (3*3)): List of the 8 gell mann matrices.

        Returns:
            void
            l_Qi (list of matrix (3*3)): List of the 8 Qi matrices.
        """
        self.Pauli_matrices()
        l_Qi_temp=[]
        for i in self.L_Gell_Mann:
            l_Qi_temp.append(1/2*i)
        self.l_Qi=l_Qi_temp

    def C_Q_i_alpha(self,i,alpha):
        """This function compute the full matrice of the operator Qi for a specific neutrino. It will compute the operator I3*...*Qi*...*I3 with * the kroneker product and Qi place at the place alpha.

        Args:
            self:
                N (int): number of neutrino
                l_Qi (list of matrix (3*3)): List of the 8 Qi matrices.
            i (int): index of the Qi we will compute (i in [1,8])
            alpha (int) index of the neutrino for which we will compute the operator Qi (alpha in [1,N])

        Returns:
            Qi_alpha (matrix): I3*...*Qi*...*I3 with * the kroneker product and Qi place at the place alpha
        """
        rest=self.N-alpha
        i=i-1#1->8
        if alpha-1==0:
            I_left=np.matrix([1])
        else:
            I_left=np.matrix(np.eye(3**(alpha-1)))
        if rest==0:
            I_right=np.matrix([1])
        else:
            I_right=np.matrix(np.eye(3**(rest)))
        return np.kron(np.kron(I_left,self.l_Qi[i]),I_right)


    def compute_H_nu(self):
        """This function compute the Hamiltonian corresponding to the one body term, coefficient already treated here.

        Args:
            self:
                The H1 parameter

        Returns:
            void
            H_nu (matrix): the matrix of the one body hamiltonian

        """
        out=self.wp*self.N*self.C_Q_i_alpha(3,1)+self.Wbp*self.N*self.C_Q_i_alpha(8,1)
        for i in range(1,self.N):
            out=out+self.wp*(self.N-i)*self.C_Q_i_alpha(3,i+1)+self.Wbp*(self.N-i)*self.C_Q_i_alpha(8,i+1)
        self.H_nu=out

    def mu_a_a2(self):
        """As the two body interraction term is time dependant we will separate the calculation of the H2 term from his coefficient to avoid recalculate everytime the H2 term, but only it's factor. In this case this function compute the time dependant function corresponding to this factor.

        Args:
            self:
                parameter corresponding to the H2 coefficient

        Returns:
            void
            self:
                mu_r_t (func t->coeff): Function that compute the H2 coefficient at a specific time t
        """
        def function_mu_R_t(t):
            return self.mu_R_nu*(1-np.sqrt(1-(self.R_mu/(t+self.starting_point_methode1))**2))**2
        self.mu_r_t=function_mu_R_t

    def compute_H_nunu(self):
        """Function that compute the H2 matrix without it's coefficient. In our case, we have the H_nunu not time dependant as we can move the time dependance outside of the matrix term.

        Args:
            self:
                parameter related to the two body interraction calculation

        Returns:
            void
            self:
                H_nunu (matrix): Matrix containing the two body interraction without it's time dependant coefficient
        """
        self.mu_a_a2()
        out=np.zeros((3**(self.N),3**(self.N)))
        for p1 in range(1,self.N+1):
            for p2 in range(1,self.N+1):
                for i in range(1,9):
                    Q_p1=self.C_Q_i_alpha(i,p1)
                    #remove diagonal term
                    if p1!=p2: #else:
                        Q_p2=self.C_Q_i_alpha(i,p2)
                        out=out+np.matmul(Q_p1,Q_p2)
        self.H_nunu=out

    def compute_H_element(self):
        """This function is part of the initialis function. It call all the function related to the initialisation of the hamiltonian. It initialise and do the calculation of each part of the Hamiltonian but without putting everything together.
        """
        #compute H_nu
        tpsa=time.perf_counter()
        self.compute_H_nu()
        tpsb=time.perf_counter()
        print("H_nu done in ",tpsb-tpsa,"s")

        #compute H_nunu
        tpsa=time.perf_counter()
        self.mu_a_a2()
        self.compute_H_nunu()
        tpsb=time.perf_counter()
        print("H_nunu done in ",tpsb-tpsa,"s")

    def sum_H(self,t=-1):#compute H for a specific t if H is time dependent
        """This function, called at each time step, will add together both Hamiltonian with their respective coefficient.

        Return:
            The whole Hamiltonian evaluate at a time t
        """
        if t==-1:
            print("You need to give a t because H is time dependant")
            return None
        else:
            return self.H_nu+1/np.sqrt(2)*self.mu_r_t(t)*self.H_nunu

    def Parameter_PMNS(self):
        """This is where we define our mixing angle coefficient needed for the calculation of the PMNS

        Args:
            self: Used to store the output

        Return:
            Void
            self:
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
            self: Used to store the output and provide the following input:
                theta_12 (float): Mixing angle between state 1 and 2 in °
                theta_23 (float): Mixing angle between state 2 and 3 in °
                theta_13 (float): Mixing angle between state 1 and 3 in °
                delta_cp (float): Charge parity violation in °

        returns:
            Void
            self:
                PMNS_element (np.matrix): 3*3 matrix containing the PMNS matrice corresponding to the qutrite encoding
        """
        c12=np.cos(self.theta_12)
        s12=np.sin(self.theta_12)
        c23=np.cos(self.theta_23)
        s23=np.sin(self.theta_23)
        c13=np.cos(self.theta_13)
        s13=np.sin(self.theta_13)
        ep_delta=np.exp(1j*self.delta_cp)
        em_delta=np.exp(-1j*self.delta_cp)

        self.PMNS_element=np.matrix([[c12*c13,s12*c13,s13*em_delta],[-s12*c23-c12*s23*s13*ep_delta,c12*c23-s12*s23*s13*ep_delta,s23*c13],[s12*s23-c12*c23*s13*ep_delta,-c12*s23-s12*c23*s13*ep_delta,c23*c13]])

    def fast_kroneker(self,x,n):#compute X*X*X*X... ntime for the kronecker product
        """Function that compute in log complexity X^n with the kroneker product as the product operation.

        Args:
            self:
            x (matrix): matrix to compute the knonecker with
            n (int): power to apply

        Returns:
            X^n with the kroneker product as the product operation
        """
        if n==0:
            return 1
        elif n%2==0:
            return self.fast_kroneker(np.kron(x,x),n/2)
        elif n%2==1:
            return np.kron(x,self.fast_kroneker(x,n-1))

    def C_PMNS(self):
        """Compute both matrix corresponding to the base changes between mass and flavor basis.

        Returns:
            self:
                PMNS_to_flavor (matrix): Matrix needed for the base changes from mass to flavor
                PMNS_to_mass (matrix): Matrix needed for the basis changes form flavor to mass
        """
        self.Parameter_PMNS()
        self.compute_PMNS_element()
        self.PMNS_to_flavor=self.fast_kroneker(self.PMNS_element,self.N)
        self.PMNS_to_mass=self.fast_kroneker(self.PMNS_element.H,self.N)

    def init_state_flavor(self):#element in 1,2 and 3
        """This is where you set the initial state in the flavor basis

        Args:
            self: Used to hold the output and provide the following input:
                N (int): Number of neutrino

        Returns:
            Void
            self:
                init_state_flavor (list of int): Initial state of the system, each state is either 0,1 or 2 correspond to each flavor state electron, muon and tau
        """
        if self.N==1:
            self.State_flavor=[0]
        elif self.N==2:
            self.State_flavor=[0,1]
        elif self.N==3:
            self.State_flavor=[0,1,2]
        elif self.N==4:
            self.State_flavor=[0,0,0,0]
        elif self.N==5:
            self.State_flavor=[0,0,0,0,0]
        elif self.N==6:
            self.State_flavor=[0,0,0,0,0,0]
        elif self.N==7:
            self.State_flavor=[0,0,0,0,0,0,0]
        else:
            print("You don't have an initial state for N=",str(self.N))
        print("The initial state in flavor state is:",self.State_flavor)

    def init_Phi0(self):
        """Compute the initial wavefunction in both mass and flavor state from the initial state.

        Args:
            self:
                State_flavor (list of int): Initial state of the system, each state is either 0,1 or 2 correspond to each flavor state electron, muon and tau
                PMNS_to_mass (matrix): Matrix needed for the basis changes form flavor to mass

        Returns:
            void
            self:
                Phi0_flavor (matrix): initial wavefunction in flavor basis
                Phi0_mass (matrix): initial wavefunction in mass basis
        """
        self.init_state_flavor()
        self.Phi0_flavor=self.state_to_phi(self.State_flavor)
        self.Phi0_mass=np.matmul(self.PMNS_to_mass,self.Phi0_flavor)
        print(np.linalg.norm(self.Phi0_flavor))
        print(np.linalg.norm(self.Phi0_mass))

    def state_to_phi(self,vect):#vect is a list with element in 0,1,2
        """This function turn a state into the corresponding wavefunction.

        Args:
            self:
            vect: (list of int): Initial state of the system, each state is either 0,1 or 2 correspond to each flavor state electron, muon and tau

        Returns:
            out (matrix): wavefunction corresponding to the initial state given "vect"
        """
        out=1
        for i in vect:
            if i==0:
                out=np.kron(np.matrix([[1],[0],[0]]),out)
                #out=np.kron(out,np.matrix([[1],[0],[0]]))
            elif i==1:
                out=np.kron(np.matrix([[0],[1],[0]]),out)
                #out=np.kron(out,np.matrix([[0],[1],[0]]))
            elif i==2:
                out=np.kron(np.matrix([[0],[0],[1]]),out)
                #out=np.kron(out,np.matrix([[0],[0],[1]]))
        #print(np.matrix(out))
        return out


    def C_l_phi_t_RK4(self,nb_point_to_keep):
        """This fonction compute the propagation over time of the Hamiltonian, for that it's using RK4 to solve the schrodinger equation.

        Args:
            self:
                Phi0_mass (matrix): initial wavefunction in mass basis
                l_t (list of float): list of all the point we will propagate through
            nb_point_to_keep (int): number of point to keep, because the time step is small we will end up with two much point that will just slow down later calculation and for memory limitation.

        Returns:
            l_phi_out (list of matrix): list of the wavefunction in mass basis at all the time kept
            self:
                l_t (list of float): the list of time as also been modified to only keep the time we keep the point in l_phi_out
                l_phi_t_mass (list of matrix): this is the exact same list as l_phi_out
                l_phi_t_flavor (list of matrix):  list of the wavefunction in flavor basis at all the time kept
        """

        print("we need to do:",self.t_nb_step)
        tps1=time.perf_counter()

        phi0=self.Phi0_mass
        lt=self.l_t
        n,m=np.shape(phi0)#m is obviously 1, vector

        def fonction_f(y,t,hbar=1):
            """Function needed for the RK4 method, we find it from the schodinger equation"""
            return -(1j/hbar)*np.sum(np.matmul(self.sum_H(t),y),1)

        def runge_kutta_4(f,y0,t,classes,nb_point_to_keep):
            """Usual implementation of the Range-Kutta 4, but we will only save some point"""
            l_point_to_keep=np.int_(np.linspace(0,classes.t_nb_step-1,nb_point_to_keep))
            l_phi_out=[y0]
            l_t_out=[t[0]]
            j=1

            n_t=len(t)
            y=y0
            tps1=time.perf_counter()
            for i in range(n_t-1):
                if i%10000==0:
                    tps2=time.perf_counter()
                    print("step:",i,tps2-tps1)
                    tps1=time.perf_counter()
                dt=t[i+1]-t[i]
                k1=f(y, t[i])
                k2=f(y+dt*k1/2.,t[i]+dt/2.)
                k3=f(y+dt*k2/2.,t[i]+dt/2.)
                k4=f(y+dt*k3,t[i]+dt)
                y=y+(dt/6.)*(k1+2.*k2+2.*k3+k4)
                if i==l_point_to_keep[j]:
                    j=j+1
                    l_phi_out.append(y)
                    l_t_out.append(t[i])

            self.l_t=l_t_out
            return l_phi_out
        l_phi_out=runge_kutta_4(fonction_f,phi0,lt,self,nb_point_to_keep)
        self.l_phi_t_mass=l_phi_out

        #convert back to flavor basis
        y_sol_flavor=[]
        for i in l_phi_out:
            y_sol_flavor.append(np.matmul(self.PMNS_to_flavor,np.matrix(i)))
        self.l_phi_t_flavor=y_sol_flavor
        tps2=time.perf_counter()
        print("RK4 is done in:",tps2-tps1,"s")
        return l_phi_out

    def format_phi_t(self):
        """This function change the format of the l_phi_t_flavor and l_phi_t_mass. It need to be used after computing the l_phi_t_flavor and l_phi_t_mass to not cause error.
        """
        n_t=len(self.l_t)
        n_case=len(self.l_phi_t_flavor[0])
        out=np.zeros((n_t,n_case+1),dtype=np.complex_)
        out2=np.zeros((n_t,n_case+1),dtype=np.complex_)
        print(np.shape(out))
        for i in range(len(self.l_t)):
            for j in range(0,n_case):
                out[i,j+1]=self.l_phi_t_flavor[i][j,0]
                out2[i,j+1]=self.l_phi_t_mass[i][j,0]
        out[:,0]=np.array(self.l_t)
        out2[:,0]=np.array(self.l_t)
        return out,out2

    def save_tab_phi_t(self,tab_flavor,tab_mass):
        """This function save both the tab_flavor and the tab_mass

        Args:
            self:
            tab_flavor (array): Formated array of the wavefunction in flavor basis
            tab_mass (array): Formated array of the wavefunction in mass basis

        Returns:
            void
            save both the file at the path "Path_3saveur"
        """
        print("saving phi_t_flavor")
        with open(self.Path_3saveur+"t_phi_t_flavor="+str(self.N)+"classical_qutrite.npy", 'wb') as f:
            np.save(f, tab_flavor)
        with open(self.Path_3saveur+"t_phi_t_mass="+str(self.N)+"classical_qutrite.npy", 'wb') as f:
            np.save(f, tab_mass)

    def load_tab_phi_t(self):
        """This function load both the tab_flavor and the tab_mass. It load files from the path "Path_3saveur".

        Args:
            self:

        Returns:
            tab_flavor (array): Formated array of the wavefunction in flavor basis
            tab_mass (array): Formated array of the wavefunction in mass basis
        """
        print("loading phi_t_flavor")
        with open(self.Path_3saveur+"t_phi_t_flavor="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            out=np.load(f)
        with open(self.Path_3saveur+"t_phi_t_mass="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            out2=np.load(f)
        return out,out2

    def compute_polarization_vector(self,tab_phi_t_mass):
        """This function compute the polarization vector in both mass and flavor state from the result of the measurement in mass state.

        Args:
            self:
            tab_phi_t_mass (array):  Formated array of the wavefunction in mass basis

        Returns:
            void
            self:
                tab_polarisation_vector (array of shape (nb_t,8,N)): array storing all the polarisation vector in mass basis. P_i,j(t)=tab_polarisation_vector[t,i,j]
                tab_P_nu1 (array of shape (nb_t,N)): array storing the population the first mass state for each neutrino and each time.
                tab_P_nu2 (array of shape (nb_t,N)): array storing the population the second mass state for each neutrino and each time.
                tab_P_nu3 (array of shape (nb_t,N)): array storing the population the third mass state for each neutrino and each time.
                tab_polarisation_vector_flavor (array of shape (nb_t,8,N)): tab storing all the polarisation vector in flavor basis. P_i,j(t)=tab_polarisation_vector_flavor[t,i,j]
                tab_P_e (array of shape (nb_t,N)): array storing the population the first flavor state for each neutrino and each time.
                tab_P_mu (array of shape (nb_t,N)): array storing the population the second flavor state for each neutrino and each time.
                tab_P_tau (array of shape (nb_t,N)): array storing the population the third flavor state for each neutrino and each time.
        """
        nb_t,nb_case=np.shape(tab_phi_t_mass)

        tab_out=np.zeros((nb_t,8,self.N))
        tab_t_nu1_temp=np.zeros((nb_t,self.N))
        tab_t_nu2_temp=np.zeros((nb_t,self.N))
        tab_t_nu3_temp=np.zeros((nb_t,self.N))

        tab_out_flavor=np.zeros((nb_t,8,self.N))
        tab_t_e_temp=np.zeros((nb_t,self.N))
        tab_t_mu_temp=np.zeros((nb_t,self.N))
        tab_t_tau_temp=np.zeros((nb_t,self.N))

        def rho_flavor_basis(class_a,rho):
            """Function that do the basis change of a matrix. We will only use it to change basis from the mass to the flavor basis.
            Args:
                class_a : class self to send
                rho (matrix): density matrice to change basis
            """
            PMNS_to_flavor=class_a.PMNS_element
            return np.matmul(np.matmul(np.conjugate(PMNS_to_flavor),rho),PMNS_to_flavor.T)

        for t in range(nb_t):
            phi=np.matrix(tab_phi_t_mass[t,1:])#(1,9)
            D_t=np.array(np.matmul(phi.H,phi))
            #print(D_t)
            for i in range(self.N):
                l=[k for k in range(self.N)]
                l.pop(i)

                rho=partial_trace(D_t,l,[3 for k in range(self.N)])
                #we need to compute rho_flavor form coefficinet of rho
                rho_flavor=rho_flavor_basis(self,rho)

                for j in range(8):
                    tab_out_flavor[t,j,i]=np.trace(np.matmul(rho_flavor,self.L_Gell_Mann[j]))
                tab_t_e_temp[t,i]=1/3*(1+3/2*tab_out_flavor[t,2,i]+np.sqrt(3)/2*tab_out_flavor[t,7,i])
                tab_t_mu_temp[t,i]=1/3*(1-3/2*tab_out_flavor[t,2,i]+np.sqrt(3)/2*tab_out_flavor[t,7,i])
                tab_t_tau_temp[t,i]=1/3*(1-np.sqrt(3)*tab_out_flavor[t,7,i])


                for j in range(8):
                    tab_out[t,j,i]=np.trace(np.matmul(rho,self.L_Gell_Mann[j]))
                tab_t_nu1_temp[t,i]=1/3*(1+3/2*tab_out[t,2,i]+np.sqrt(3)/2*tab_out[t,7,i])
                tab_t_nu2_temp[t,i]=1/3*(1-3/2*tab_out[t,2,i]+np.sqrt(3)/2*tab_out[t,7,i])
                tab_t_nu3_temp[t,i]=1/3*(1-np.sqrt(3)*tab_out[t,7,i])
        self.tab_polarisation_vector=tab_out
        self.tab_P_nu1=tab_t_nu1_temp
        self.tab_P_nu2=tab_t_nu2_temp
        self.tab_P_nu3=tab_t_nu3_temp

        self.tab_polarisation_vector_flavor=tab_out_flavor
        self.tab_P_e=tab_t_e_temp
        self.tab_P_mu=tab_t_mu_temp
        self.tab_P_tau=tab_t_tau_temp


    def save_polarization_vector(self):
        """save all the array related to the polarization vector to the path "Path_3saveur".

        Args:
            self:
                tab_polarisation_vector (array of shape (nb_t,8,N)): array storing all the polarisation vector in mass basis. P_i,j(t)=tab_polarisation_vector[t,i,j]
                tab_P_nu1 (array of shape (nb_t,N)): array storing the population the first mass state for each neutrino and each time.
                tab_P_nu2 (array of shape (nb_t,N)): array storing the population the second mass state for each neutrino and each time.
                tab_P_nu3 (array of shape (nb_t,N)): array storing the population the third mass state for each neutrino and each time.
                tab_polarisation_vector_flavor (array of shape (nb_t,8,N)): tab storing all the polarisation vector in flavor basis. P_i,j(t)=tab_polarisation_vector_flavor[t,i,j]
                tab_P_e (array of shape (nb_t,N)): array storing the population the first flavor state for each neutrino and each time.
                tab_P_mu (array of shape (nb_t,N)): array storing the population the second flavor state for each neutrino and each time.
                tab_P_tau (array of shape (nb_t,N)): array storing the population the third flavor state for each neutrino and each time.

        Returns:
            void
            save all the args file to the path "Path_3saveur".
        """
        print("saving polarization vector")
        with open(self.Path_3saveur+"tab_Polarization_vector="+str(self.N)+"classical_qutrite.npy", 'wb') as f:
            np.save(f, self.tab_polarisation_vector)
        with open(self.Path_3saveur+"tab_P_nu1="+str(self.N)+"classical_qutrite.npy", 'wb') as f:
            np.save(f, self.tab_P_nu1)
        with open(self.Path_3saveur+"tab_P_nu2="+str(self.N)+"classical_qutrite.npy", 'wb') as f:
            np.save(f, self.tab_P_nu2)
        with open(self.Path_3saveur+"tab_P_nu3="+str(self.N)+"classical_qutrite.npy", 'wb') as f:
            np.save(f, self.tab_P_nu3)

        with open(self.Path_3saveur+"tab_polarisation_vector_flavor="+str(self.N)+"classical_qutrite.npy", 'wb') as f:
            np.save(f, self.tab_polarisation_vector_flavor)
        with open(self.Path_3saveur+"tab_P_e="+str(self.N)+"classical_qutrite.npy", 'wb') as f:
            np.save(f, self.tab_P_e)
        with open(self.Path_3saveur+"tab_P_mu="+str(self.N)+"classical_qutrite.npy", 'wb') as f:
            np.save(f, self.tab_P_mu)
        with open(self.Path_3saveur+"tab_P_tau="+str(self.N)+"classical_qutrite.npy", 'wb') as f:
            np.save(f, self.tab_P_tau)


    def load_polarization_vector(self):
        """loas all the array related to the polarization vector to the path "Path_3saveur".

        Args:
            self:

        Returns:
            void
            self:
                tab_polarisation_vector (array of shape (nb_t,8,N)): array storing all the polarisation vector in mass basis. P_i,j(t)=tab_polarisation_vector[t,i,j]
                tab_P_nu1 (array of shape (nb_t,N)): array storing the population the first mass state for each neutrino and each time.
                tab_P_nu2 (array of shape (nb_t,N)): array storing the population the second mass state for each neutrino and each time.
                tab_P_nu3 (array of shape (nb_t,N)): array storing the population the third mass state for each neutrino and each time.
                tab_polarisation_vector_flavor (array of shape (nb_t,8,N)): tab storing all the polarisation vector in flavor basis. P_i,j(t)=tab_polarisation_vector_flavor[t,i,j]
                tab_P_e (array of shape (nb_t,N)): array storing the population the first flavor state for each neutrino and each time.
                tab_P_mu (array of shape (nb_t,N)): array storing the population the second flavor state for each neutrino and each time.
                tab_P_tau (array of shape (nb_t,N)): array storing the population the third flavor state for each neutrino and each time.
        """
        print("loading polarization vector")
        with open(self.Path_3saveur+"tab_Polarization_vector="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            self.tab_polarisation_vector=np.load(f)
        with open(self.Path_3saveur+"tab_P_nu1="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            self.tab_P_nu1=np.load(f)
        with open(self.Path_3saveur+"tab_P_nu2="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            self.tab_P_nu2=np.load(f)
        with open(self.Path_3saveur+"tab_P_nu3="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            self.tab_P_nu3=np.load(f)

        with open(self.Path_3saveur+"tab_polarisation_vector_flavor="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            self.tab_polarisation_vector_flavor=np.load(f)
        with open(self.Path_3saveur+"tab_P_e="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            self.tab_P_e=np.load(f)
        with open(self.Path_3saveur+"tab_P_mu="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            self.tab_P_mu=np.load(f)
        with open(self.Path_3saveur+"tab_P_tau="+str(self.N)+"classical_qutrite.npy", 'rb') as f:
            self.tab_P_tau=np.load(f)
##Plot function

def plot_polarization_vector(tab_nu1,tab_nu2,tab_nu3,tab_polarization_vector,l_t,which_to_plot):
    """Function that Plot the Polarization vector corresponding graph based on the argument which_to_plot.

    Args:
        tab_nu1 (array): Containt the first state either mass or flavor for different time and different neutrino.
        tab_nu2 (array): Containt the second state either mass or flavor for different time and different neutrino.
        tab_nu3 (array): Containt the third state either mass or flavor for different time and different neutrino.
        tab_polarization_vector (array): contain the 8 polarization vector for different time and all the neutrino
        l_t (list): list of time associated to these array
        which_to_plot (int): id to tell which graph to plot:
            [0-7]->P1-P8
            -1->first state
            -2->second state
            -3->third state

    Returns:
        void
        plot the result
    """
    #tab_polarization_vector[t,i,j] for Pi of neutrino j at t
    #tab_nu1[t,j] for P_nu1(t) of neutrino j
    #which_to_plot int [-3,7] positive for tab_polarization and negative for tab_nu
    def plot(l_label,l_x,L_l_y):
        n_plot=len(L_l_y)
        l_style=["-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":"]
        for i in range(n_plot):
            plt.plot(l_x+210.64,L_l_y[i],label=l_label[i],linestyle=l_style[i])
        plt.xlabel("$\kappa^{-1}$")
        plt.legend()
        #plt.xscale("log")
        plt.show()

    def construct_L_l_y_nu(tab,j=-1):
        L_l_out=[]
        l_label=[]
        if j<0:#tab_nu1
            n_t,N=np.shape(tab)
            for i in range(N):
                L_l_out.append(tab[:,i])
                l_label.append("Pnu_"+str(-1*j)+" N="+str(i))
                plt.ylim(ymin=0,ymax=1)
                plt.ylabel("Probability")

        else:#tab_polarization_vector
            n_t,known,N=np.shape(tab)
            for i in range(N):
                L_l_out.append(tab[:,j,i])
                l_label.append("P_"+str(j)+" N="+str(i))
                plt.ylim(ymin=-1,ymax=1)
        return L_l_out,l_label

    if which_to_plot==-1:
        L_l_y,l_label=construct_L_l_y_nu(tab_nu1,-1)
    elif which_to_plot==-2:
        L_l_y,l_label=construct_L_l_y_nu(tab_nu2,-2)
    elif which_to_plot==-3:
        L_l_y,l_label=construct_L_l_y_nu(tab_nu3,-3)
    elif which_to_plot>=0:
        L_l_y,l_label=construct_L_l_y_nu(tab_polarization_vector,which_to_plot)
    plot(l_label,l_t,L_l_y)

def plot_proba_state(tab_P_e,tab_P_mu,tab_P_tau,l_t,bool_flavor=True):
    """Function that plot the population of the three state for a single neutrino.

    Args:
        tab_P_e (array): Containt the first state either mass or flavor for different time and different neutrino.
        tab_P_mu (array): Containt the second state either mass or flavor for different time and different neutrino.
        tab_P_tau (array): Containt the third state either mass or flavor for different time and different neutrino.
        bool_flavor (boolean): True if in flavor state, False otherwise

    Returns:
        void
        plot the figure
    """
    l_style=["-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":"]
    if bool_flavor:
        plt.plot(l_t+210.64,tab_P_e[:,0],label="$n_{e}$",linestyle=l_style[0])
        plt.plot(l_t+210.64,tab_P_mu[:,0],label="$n_{\mu}$",linestyle=l_style[1])
        plt.plot(l_t+210.64,tab_P_tau[:,0],label="$n_{tau}$",linestyle=l_style[2])
    else:
        plt.plot(l_t+210.64,tab_P_e[:,0],label="$n_{1}$",linestyle=l_style[0])
        plt.plot(l_t+210.64,tab_P_mu[:,0],label="$n_{2}$",linestyle=l_style[1])
        plt.plot(l_t+210.64,tab_P_tau[:,0],label="$n_{3}$",linestyle=l_style[2])
    plt.xlabel("$\kappa^{-1}$",fontsize=18)
    plt.ylabel("Probability",fontsize=18)
    plt.legend()
    #plt.xscale("log")
    plt.show()

def multiplot_pnu_state(tab_P_e,tab_P_mu,tab_P_tau,l_t,bool_flavor=True):
    """This function plot the figure composed of three panel. In each we plot the evolution over time for each neutrino of one of the three state. Depending on the input either flavor or mass state.

    Args:
        tab_P_e (array): Containt the first state either mass or flavor for different time and different neutrino.
        tab_P_mu (array): Containt the second state either mass or flavor for different time and different neutrino.
        tab_P_tau (array): Containt the third state either mass or flavor for different time and different neutrino.
        bool_flavor (boolean): True if in flavor state, False otherwise

    Returns:
        void
        plot the figure
    """
    l_color=list(mcolors.TABLEAU_COLORS.values())
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    n_t,n_neutrino=np.shape(tab_P_e)
    for i in range(n_neutrino):
        axs[0].plot(l_t,tab_P_e[:,i],l_color[i],linestyle="-",label="neutrino "+str(i))

        axs[1].plot(l_t,tab_P_mu[:,i],l_color[i],linestyle="-",label="neutrino "+str(i))

        axs[2].plot(l_t,tab_P_tau[:,i],l_color[i],linestyle="-",label="neutrino "+str(i))

    axs[-1].set_xlabel("t($\kappa ^{-1}$)", fontsize=18)
    if bool_flavor:
        axs[0].set_ylabel("$P_{e}$")
        axs[1].set_ylabel("$P_{\mu}$")
        axs[2].set_ylabel("$P_{\tau}$")
    else:
        axs[0].set_ylabel("$P_{1}$")
        axs[1].set_ylabel("$P_{2}$")
        axs[2].set_ylabel("$P_{3}$")
    """
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    axs[2].set_xscale('log')
    """
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    plt.show()

def tab_fusion_asked(tab_P_nu1,tab_P_nu2,tab_P_nu3,tab_P_e,tab_P_mu,tab_P_tau,l_t):
    """This figure is composed of two plot. At the top we plot the time propagation of the population of the three flavor state for one neutrino. At the bottom we plot the time propagation of the population of the three mass state for one neutrino.

    Args:
        tab_P_nu1 (array): Containt the first mass state for different time and different neutrino.
        tab_P_nu2 (array): Containt the second mass state for different time and different neutrino.
        tab_P_nu3 (array): Containt the third mass state for different time and different neutrino.
        tab_P_e (array): Containt the first flavor state for different time and different neutrino.
        tab_P_mu (array): Containt the second flavor state for different time and different neutrino.
        tab_P_tau (array): Containt the third flavor state for different time and different neutrino.
        l_t (list of float): List of corresponding time points

    Returns:
        void
        Plot the figure
    """
    l_style=["-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":"]
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)


    axs[0].plot(l_t+210.64,tab_P_e[:,0],label="$n_{e}$",linestyle=l_style[0])
    axs[0].plot(l_t+210.64,tab_P_mu[:,0],label="$n_{\mu}$",linestyle=l_style[1])
    axs[0].plot(l_t+210.64,tab_P_tau[:,0],label="$n_{tau}$",linestyle=l_style[2])

    axs[1].plot(l_t+210.64,tab_P_nu1[:,0],label="$n_{1}$",linestyle=l_style[0])
    axs[1].plot(l_t+210.64,tab_P_nu2[:,0],label="$n_{2}$",linestyle=l_style[1])
    axs[1].plot(l_t+210.64,tab_P_nu3[:,0],label="$n_{3}$",linestyle=l_style[2])

    axs[0].set_xlabel("$\kappa^{-1}$")
    axs[0].set_ylabel("Probability")
    axs[1].set_ylabel("Probability")
    axs[0].legend()
    axs[1].legend()
    #plt.xscale("log")
    plt.show()

##other
def compute_commutator(l_qi):
    """This function was used to ease and verify the calculation of commutator of Qi. It was needed to verify the frequency of neutrino oscillation.
    """
    def commutator(X,Y):
        """Compute the commutation of two matrix."""
        return np.matmul(X,Y)-np.matmul(Y,X)
    L_l_out=[]
    for i in range(8):
        L_collumns=[]
        for j in range(8):
            L_collumns.append(commutator(l_qi[i],l_qi[j]))
        L_l_out.append(L_collumns)
    return L_l_out

def convert_t():#fonction that use k^-1 and convert it into into km/GeV unit
    """
    """
    q=10**(17)              #MeV-1
    q=q*10**3               #GeV-1          #convert to GeV
    q=q*0.197*10**(-15)     #m              #convert to meter
    q=q*10**(-3)            #Km             #convert to km
    #print(q)
    q=q/(10*10**(-3))       #km/GeV         #divide by E


    print("We convert  1 time unit in k^-1 in MeV^{-1} to:",q,"in Km/GeV.")
    T=17#measured time periode of an oscillation in \kappa^-1

    print("We measured a periode of oscillation as:",T,"k^-1")
    print("Periode of an oscillation", q*17,"in Km/GeV")
    #return q #in Km/GeV

##Application
"""This section contain some already prepared code to directly use the function presented ealier.

Before executing the code, you need to check the path at which data will save and load from.
    - The "Path_3saveur" correspond to the result of this code

In order to use them you can interract with the following parameter:
-initial state written in the initialise_state_flavor method
-mixing angle written in the Parameter_PMNS method
-number of neutrino by changing the value of n_neutrino below or when calling Hamiltionian_classique_SU3(n)

-Rest of the coefficient link to the simulation: __init__
    * t_start : time at which we start doing the calculation
    * t_finish : time a which we finish doing the calulation
    * dt : time step to use
    * starting_point_methode1 : phase difference of the interraction term. A too low value may diverge the calculation
    * parameter of the U1 and U2 propagator

If you want to modify hamiltonian:
-compute_H_nu controle the one body part of the Hamiltonian (it also contain it's coefficient)
-comput_H_nunu controle the interraction term of the Hamiltonian. (it doesn't contain it's coefficient
-mu_a_a2 controle the coefficient of the interraction term (it return a time dependant function
-sum_H is called in RK4 methode at each time step and add all the part of the Hamiltonian for a specific time.

Execution parameter:
    * m controle the task we are going to do
    * n controle the number of neutrino
    * nb_point_to_keep controle the number of point we are going to keep (independantly of the number of point used to do the propagation)
    * i parameter used for the plot_polarization_vector function, correspond to it's which_to_plot parameter


The m parameter control which work you want to do:

* m==0: Compute the propagation in both mass and flavor state and save the wavefunction.
* m==1: Load the wavefunction and compute the polarization vector as well as the population of each mass and flavor state. Finally saved them.
* m==3: Do the m==0 and m==1 processus and save everything. The whole calculation is done.
* m==4: Load the data and plot the following figure in mass basis:
    - plot_polarization_vector for the i parameter
    - plot_proba_state
    - multiplot_pnu_state
* m==5: Load and plot the following figure in flavor basis:
    - plot_polarization_vector for the i parameter
    - plot_proba_state
    - multiplot_pnu_state
* m==6: Load and plot the following figure in both basis:
    - tab_fusion_asked

If you want to remove the interraction term you have to remove it by hand in the sum_H method.
"""
m=6

n=3
nb_point_to_keep=10000
i=-2#in [-3,7]

if m==0:#just compute propagation phi_t
    a=Hamiltionian_classique_SU3(n)
    print(np.shape(a.H_nu))

    a.C_l_phi_t_RK4(nb_point_to_keep)
    tab_phi_t_flavor,tab_phi_t_mass=a.format_phi_t()

    a.save_tab_phi_t(tab_phi_t_flavor,tab_phi_t_mass)

elif m==1: #just compute polarization vector
    a=Hamiltionian_classique_SU3(n)
    print(np.shape(a.H_nu))

    tab_phi_t_flavor,tab_phi_t_mass=a.load_tab_phi_t()

    a.compute_polarization_vector(tab_phi_t_mass)
    a.save_polarization_vector()

elif m==3:#compute both propagation and polarization vector
    a=Hamiltionian_classique_SU3(n)
    print(np.shape(a.H_nu))

    a.C_l_phi_t_RK4(nb_point_to_keep)
    tab_phi_t_flavor,tab_phi_t_mass=a.format_phi_t()
    a.save_tab_phi_t(tab_phi_t_flavor,tab_phi_t_mass)
    a.compute_polarization_vector(tab_phi_t_mass)
    a.save_polarization_vector()

elif m==4: #plot polarisation mass basis

    a=Hamiltionian_classique_SU3(n)
    print(np.shape(a.H_nu))
    a.load_polarization_vector()
    tab_phi_t_flavor,tab_phi_t_mass=a.load_tab_phi_t()
    plot_polarization_vector(a.tab_P_nu1,a.tab_P_nu2,a.tab_P_nu3,a.tab_polarisation_vector,tab_phi_t_mass[:,0],i)
    plot_proba_state(a.tab_P_nu1,a.tab_P_nu2,a.tab_P_nu3,tab_phi_t_mass[:,0],False) #plot for neutrino 0 of the masses over time
    multiplot_pnu_state(a.tab_P_nu1,a.tab_P_nu2,a.tab_P_nu3,tab_phi_t_mass[:,0],False)
elif m==5: #plot polarisation flavor basis

    a=Hamiltionian_classique_SU3(n)
    print(np.shape(a.H_nu))
    a.load_polarization_vector()
    tab_phi_t_flavor,tab_phi_t_mass=a.load_tab_phi_t()
    plot_polarization_vector(a.tab_P_e,a.tab_P_mu,a.tab_P_tau,a.tab_polarisation_vector_flavor,tab_phi_t_mass[:,0],i)
    plot_proba_state(a.tab_P_e,a.tab_P_mu,a.tab_P_tau,tab_phi_t_mass[:,0])
    multiplot_pnu_state(a.tab_P_e,a.tab_P_mu,a.tab_P_tau,tab_phi_t_mass[:,0])

elif m==6:
    a=Hamiltionian_classique_SU3(n)
    print(np.shape(a.H_nu))
    a.load_polarization_vector()
    tab_phi_t_flavor,tab_phi_t_mass=a.load_tab_phi_t()
    tab_fusion_asked(a.tab_P_nu1,a.tab_P_nu2,a.tab_P_nu3,a.tab_P_e,a.tab_P_mu,a.tab_P_tau,tab_phi_t_mass[:,0])


