"""This code is tasked to compute classicaly (in opposition to the quantum computation) the two flavor approximation of the oscillation of neutrino with the two body interraction.
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
"""

#import sympy
import numpy as np

import matplotlib.pyplot as plt
import scipy
import scipy.constants

from scipy.linalg import expm,logm

import time

##Construction of the Hamiltonian
"""This section is related to compute the matrix form of the Hamiltonian
"""

def XYZ_i(n_dim):
    """This fonction compute the X_i, Y_i and Z_i operator from X,Y and Z matrices.

    Args:
        n_dim (int): Number of neutrino in the system

    Returns:
        L_Xi (list of matrix): list storing the Xi for each neutrino
        L_Yi (list of matrix): list storing the Yi for each neutrino
        L_Zi (list of matrix: list storing the Zi for each neutrino
    """
    Id=np.eye(2)
    SigmaX=np.matrix([[0,1],[1,0]])
    SigmaY=np.matrix([[0,-1j],[ 1j,0]])
    SigmaZ=np.matrix([[1,0],[0,-1]])
    L_Xi=[]
    L_Yi=[]
    L_Zi=[]
    for i in range(n_dim):
        Xi,Yi,Zi=Id,Id,Id
        if i==0:
            Xi,Yi,Zi=SigmaX,SigmaY,SigmaZ
        for j in range(n_dim-1):
            if i-1==j and i!=0:
                Xi=np.kron(Xi,SigmaX)
                Yi=np.kron(Yi,SigmaY)
                Zi=np.kron(Zi,SigmaZ)
            else:
                Xi=np.kron(Xi,Id)
                Yi=np.kron(Yi,Id)
                Zi=np.kron(Zi,Id)
        L_Xi.append(Xi)
        L_Yi.append(Yi)
        L_Zi.append(Zi)
    return (L_Xi,L_Yi,L_Zi)

def C_H_nu(l_X,l_Y,l_Z,b):
    """This function compute the one body Hamiltonian

    Args:
        l_X (list of matrix): list storing the Xi for each neutrino
        l_Y (list of matrix): list storing the Yi for each neutrino
        l_Z (list of matrix: list storing the Zi for each neutrino
        b (list of float): list of the coefficient of the one body term

    Returns:
        The matrix corresponding to the one body Hamiltonian
    """
    X,Y,Z=l_X[0],l_Y[0],l_Z[0]
    for i in range(1,len(l_X)):
        X=X+l_X[i]
        Y=Y+l_Y[i]
        Z=Z+l_Z[i]
    return b[0]*X+b[1]*Y+b[2]*Z

def C_H_nunu(l_X,l_Y,l_Z,J):
    """This function compute the two body interraction part of the hamiltonian.

    Args:
        l_X (list of matrix): list storing the Xi for each neutrino
        l_Y (list of matrix): list storing the Yi for each neutrino
        l_Z (list of matrix: list storing the Zi for each neutrino
        J (matrix): J give the coefficient of the two body interraction term of the Hamilonian giving. J[i,j] is the term to use when looking at the interraction between neutrino i and neutrino j

    Returns:
        S (matrix): matrix corresponding to the two body interraction term of the Hamiltonian
    """
    n,m=np.shape(l_X[0])
    S=np.zeros((n,n))
    for i in range(len(l_X)):
        for j in range(i+1,len(l_X)):
            S=S+J[i,j]*(l_X[i]*l_X[j]+l_Y[i]*l_Y[j]+l_Z[i]*l_Z[j])
    return S

def C_H_n(n,J,b):#C is delta_m**2/4E
    """This function sum all the pieces of the Hamiltonian together before returning it.

    Args:
        n (int): Number of neutrino in the system
        J (matrix): J give the coefficient of the two body interraction term of the Hamilonian giving. J[i,j] is the term to use when looking at the interraction between neutrino i and neutrino j
        b (list of float): list of the coefficient of the one body term

    Returns:
        The matrix forme of the Hamiltonian
    """
    l_X,l_Y,l_Z=XYZ_i(n)

    H_nu1=C_H_nu(l_X,l_Y,l_Z,b)

    H_nunu1=C_H_nunu(l_X,l_Y,l_Z,J)

    return H_nu1+H_nunu1,l_X,l_Y,l_Z

def Jij(n,C):
    """This function compute the matrix giving the coefficient of the two body interraction term of the Hamiltonian

    Args:
        n (int): the number of neutrino we consider
        C (float): coefficient of the two body interraction term coefficient

    Returns:
        J (matrix): Jij give the coefficient of the two body interraction term of the Hamilonian giving. Jij[i,j] is the term to use when looking at the interraction between neutrino i and neutrino j
    """
    n_max=n
    J=np.zeros((n_max,n_max))
    for i in range(n_max):
        for j in range(n_max):
            J[i,j]=C*(1-np.cos(np.arccos(0.9)*np.abs(i-j)/(n-1)))
    return J

##methode 1: exact solution with exponential
"""This is the first resolution method of the schrodinger equation given this Hamiltonian
This solution is just using the analytical solution for a non time dependant Hamiltonian
"""

def C_l_phi_t(H_tot,hbar,l_t,phi0):
    """This function compute the analytical solution of the Schrodinger equation for multiple time. It return the list of the wavefunction for each time.

    Args:
        H_tot (matrix): the Hamiltonian we use in the schrodinger equation
        hbar (float): hbar, in the calculation we only used hbar=1
        l_t (list of float): list containing all the time point we want to evaluate
        phi0 (matrix/ vector): initial wavefunction at t=0

    Returns:
        A (list of matrix/vector): list of the wavefunction at each time point from l_t
    """
    tps1=time.perf_counter()
    print("ici")
    print(np.shape(H_tot))
    print(np.shape(phi0))
    A=[np.matmul(expm(-1j*t*H_tot/hbar),phi0) for t in l_t]
    tps2=time.perf_counter()

    print("exact methode done in:",tps2-tps1,"s")
    return A

##methode 2:
"""This section give another methode to solve the Hamiltonian. Instead of computing the exponential of a matrix, we will diagonalise it before computing the exponenential.
"""

def diagonalise_H_m2(H):
    """This fonction compute the eigenvalues and the eigenvactor of the hamiltonian"""
    eigenvalues, eigenvectors = np.linalg.eig(H)
    return eigenvalues, eigenvectors

def cnt(phi_0,eigenvalues,eigenvectors,t,h):
    """This function compute the new coefficient of the wavefunction for a single time point t (analytic solution base on the value of the eigenvalue and the eigenvector)

    Args:
        phi0 (matrix/ vector): initial wavefunction at t=0
        eigenvalues (list of float): eingenvalues
        eigenvectors (list of vector): eigenvectors
        t (float) time at which we want to know the wavefunction
        h (float): hbar

    Returns:
        c_alpha_t (vector): The value of the wave function at the time t
    """
    n,m=np.shape(phi_0) #m is obviously 1
    n_base=np.matrix(np.eye(n))
    eigenvectors2=np.matrix(eigenvectors)
    c_alpha_t=np.matrix(np.zeros([n,m],dtype = "complex_"))
    for j in range(n):
        res=0
        res2=0
        for alpha in range(n):
            for i in range(n):
                res=res+phi0[i]*np.dot(n_base[:,j].H,eigenvectors2[:,alpha])*np.dot(eigenvectors2[:,alpha].H,n_base[:,i])
            res2=res2+res*np.exp(-1j*eigenvalues[alpha]*t/h)
        c_alpha_t[j]=res2
    return c_alpha_t


##methode 3: Euler method
"""This is yet another methode to solve the schrodinger equation, in this case we use the euler method.
"""

def cnt3(phi_t1,dt,h,Hamiltonian): #compute the phi(t)
    """Compute one step of the euler methode, for a step of size dt.

    Args:
        phi_t1 (vector): wavefunction at point t
        dt (float): time step we want to compute
        h (float): hbar
        Hamiltonian (matrix): Full Hamiltonian

    Returns:
        phi_t2 (vector): wavefunction at point t+dt
    """
    n,m=np.shape(phi_t1) #m is obviously 1
    phi_t2=np.matrix(np.zeros([n,m],dtype = "complex_"))
    for i in range(n):
        res=0
        for j in range(n):
            res+=phi_t1[j,0]*Hamiltonian[i,j]
        phi_t2[i,0]=phi_t1[i,0]-(1j*dt/h)*res
    return phi_t2

def propagate_time(phi_0,ti,h,Hamiltonian):
    """This function call the cnt3 function and make it propagate through all the time step needed.

    Args:
        phi0 (matrix/ vector): initial wavefunction at t=0
        ti (list of float): list containing all the time point we want to evaluate
        h (float): hbar
        Hamiltonian (matrix: Full Hamiltonian
    """
    tps1=time.perf_counter()
    l_phi_t_eval=[phi_0 for ti2 in ti]
    for i in range(1,(len(ti))):
        l_phi_t_eval[i]=cnt3(l_phi_t_eval[i-1],ti[i]-ti[i-1],h,Hamiltonian)
    tps2=time.perf_counter()
    print("Euler is done in:",tps2-tps1,"s")
    return l_phi_t_eval

##methode 4: Runge kutta 2 and 4
"""Function related to Runge-Kutta 2 and 4
"""
def cnt_w_RK2(phi_0,l_t,h,Hamiltonian):
    """This function do the whole Runge-Kutta 2 process.

    Args:
        phi0 (matrix/ vector): initial wavefunction at t=0
        l_t (list of float): list containing all the time point we want to evaluate
        h (float): hbar
        Hamiltonian (matrix: Full Hamiltonian

    Returns:
        y_sol (list of vector): list of all the wavefunction at the different time given in l_t.
    """
    tps1=time.perf_counter()
    n,m=np.shape(phi_0)#m is obviously 1

    def f(y,t):
        """Function corresponding to the schrodinger equation problem given our Hamiltonian

        Args:
            y (vector): The wavefunction at time t
            t (float): the time at which we are working
        """
        f_ty=np.matrix(np.zeros([n,m],dtype = "complex_"))
        for i in range(n):
            res=0
            for j in range(n):
                #res+=phi_t1[j,0]*np.dot(np.dot(n_base[:,i].H,Hamiltonian),n_base[:,j])
                res+=y[j,0]*Hamiltonian[i,j]
            f_ty[i]=-(1j/h)*res
        return f_ty

    def rungekutta2(f, y0, t):
        """The Runge-Kutta 2 method

        Args:
            f (function): function corresponding to the schrodinger equation problem
            y0 (vector): initial point of the Runge-Kutta 2 method
            t (list of float): list of all the time point

        Returns:
            y (list of vector): list of the wavefunction at the time given in t
        """
        n = len(t)
        y = [y0]
        for i in range(n - 1):
            h = t[i+1] - t[i]
            y.append(y[i] + h * f(y[i] + f(y[i], t[i]) * h / 2., t[i] + h / 2.))
        return y
    y0 = phi_0

    y_sol=rungekutta2(f,y0,l_t)
    tps2=time.perf_counter()
    print("RK2 is done in:",tps2-tps1,"s")
    return y_sol

##operator evaluation
"""This section hold the functions that evaluate the operator from the wavefunction
"""

def Z_eval(Z,phi_t_eval):
    """This function evaluate an operator

    Args:
        Z (matrix): the operator we want to evaluate
        phi_t_eval (vecotr): the wavefunction at a specific time point

    Returns:
        (float): the value of the operator at the same time point as the wavefunction
    """
    return np.matmul(np.matmul(phi_t_eval.H,Z),phi_t_eval)

#Evaluate the population on the Z operator for all the phi(t) from the list l_phi(t) of phi(t) at different time
def l_Z_eval(l_Z,l_phi_t_eval):
    """Evaluate the population of each of the Zi operator where i is the index on the neutrino id

    Args:
        l_Z (list of matrix): list of operator corresponding to the Zi operator for different neutrino
        l_phi_t_eval (list of vector): list of the wavefunction at different time points

    Returns:
        l_Z_eval_out (list list float): The value of each Z_i for each time point. l_Z_eval_out[i][t] give the value of the of the operator Z_i at the time t
    """
    l_Z_eval_out=[]
    for i in range(len(l_Z)):
        l_Z_eval_out.append([Z_eval(l_Z[i],j)[0,0] for j in l_phi_t_eval])
    print("dim Zi:",np.shape(l_Z_eval_out))

    return(l_Z_eval_out)

#function that evaluate all the operator l_Z on the list l_Z at all the phi_t on the l_phi_t_eval and return  a list of size len(l_Z) where each element correspond to the list of the operator evaluate at each time
#actually work for operator that are not Z specifically
def evaluate_operator_Z(l_Z,l_phi_t_eval):
    """Wrap the operator measurement with some print and time measurement.

    Args:
        l_Z (list of matrix): list of operator corresponding to the Zi operator for different neutrino
        l_phi_t_eval (list of vector): list of the wavefunction at different time points

    Returns:
        Zi (list list float): The value of each Z_i for each time point. Zi[i][t] give the value of the of the operator Z_i at the time t
    """
    print("calcule of operator Z:")
    tps1=time.perf_counter()
    Zi=l_Z_eval(l_Z,l_phi_t_eval)
    tps2=time.perf_counter()
    print("calcule of operator Z is done in",tps2-tps1,"s")
    return Zi

##density/ entropy methode:
"""This section is related to the computation of the density matrice and the partial trace.
Warning, we also use the partial trace introduce at the begining of the file because it's faster.
"""

def C_density(L_phi):
    """This function compute the density matrix of the wavefunction at all time point.

    Args:
        L_phi (list of vector): list of the wavefunction at different time points

    Returns:
        D (list of matrix): list of the density matrix at different time points
    """
    D=[np.matmul(i,i.H) for i in L_phi]
    return D

def partial_trace_2_element(L_phi,id1,id2):
    """This function compute the bipartite partial trace on two elements (end up with a 4*4 matrix).

    Args:
        L_phi (list of vector): list of the wavefunction at different time points
        id1 (int): id of the first neutrino to trace on
        id2 (int): id of the seconc neutrino to trace on


    Returns:
        L_res (list of 4*4 matrix): Partial trace of the density matrix on 2 element id1 and id2 at different time points
    """

    if id1>id2:
        a=id1
        id1=id2
        id2=a
    if id1==id2:
        print("error, id1=id2")
    tpstart=time.perf_counter()
    print("compute Tr_",id1,",",id2,":")
    D=C_density(L_phi)
    n=int(np.log(len(L_phi[0]))/np.log(2))#int(np.sqrt(len(L_phi[0])))#
    print(n)

    #structure a b c d e with both b and d of size 1, a,c and e can be of size 0
    #replace n_l and n_r
    n_a=id1-1 #element before b (size of a)
    n_c=id2-id1-1 #element between b and d (size of c)
    n_e=n-id2 #element after d (size of e)

    p_na=2**n_a
    p_nc=2**n_c
    p_ne=2**n_e
    identity=np.matrix(np.eye(2))
    identity4=np.matrix(np.eye(4))

    A=np.eye(2**(n-2))#matrice of all |i>x|j> exept id1 and id2

    #el_left[i] D el_right[i] for all possible i

    if n_c==0:#|...,id1,id2,...>#Same case as a one body but Identity in 4d instead of identity in 2d
        if n_e==0: #|. . . id_element> id_element is the last element of the vector
            L_i=[np.transpose(np.matrix([A[i,:]])) for i in range(2**(n-2))]
            el_right=[np.kron(i,identity4) for i in L_i]
            el_left=[np.kron(np.transpose(i),identity4) for i in L_i]
        elif n_a==0:#|id_element . . .> id_element is the first element of the vector
            L_i=[np.transpose(np.matrix([A[i,:]])) for i in range(2**(n-2))]
            el_right=[np.kron(identity4,i) for i in L_i]
            el_left=[np.kron(identity4,np.transpose(i)) for i in L_i]
        else: #|. . id_element . .> is in the bulk of the vector
            L_i_r=[np.transpose(np.matrix([A[i,:p_ne]])) for i in range(p_ne)]
            L_i_l=[np.transpose(np.matrix([A[i,:p_na]])) for i in range(p_na)]
            el_right_r=[np.kron(identity4,i) for i in L_i_r]
            el_left_r=[np.kron(identity4,np.transpose(i)) for i in L_i_r]
            el_right=[]
            el_left=[]
            for i in range(len(el_right_r)):
                for j in range(len(L_i_l)):
                    el_right.append(np.kron(L_i_l[j],el_right_r[i]))
                    el_left.append(np.kron(np.transpose(L_i_l[j]),el_left_r[i]))
    #n_c != 0:
    elif n_a==0:#|id1,..,id2,?>
        if n_e==0:#|id1,...,id2>
            L_i=[np.transpose(np.matrix([A[i,:]])) for i in range(2**(n-2))]
            el_right=[np.kron(identity,np.kron(i,identity)) for i in L_i]
            el_left=[np.kron(identity,np.kron(np.transpose(i),identity)) for i in L_i]
        else:#|id1,...,id2,...>
            L_i_r=[np.transpose(np.matrix([A[i,:p_ne]])) for i in range(p_ne)]
            L_i_l=[np.transpose(np.matrix([A[i,:p_nc]])) for i in range(p_nc)]
            el_right_r=[np.kron(identity,i) for i in L_i_r]
            el_left_r=[np.kron(identity,np.transpose(i)) for i in L_i_r]
            el_right=[]
            el_left=[]
            for i in range(len(el_right_r)):
                for j in range(len(L_i_l)):
                    el_right.append(np.kron(identity,np.kron(L_i_l[j],el_right_r[i])))
                    el_left.append(np.kron(identity,np.kron(np.transpose(L_i_l[j]),el_left_r[i])))
    elif n_e==0:#|...,id1,...,id2> #we know for sure n_a!=0
        L_i_r=[np.transpose(np.matrix([A[i,:p_nc]])) for i in range(p_nc)]
        L_i_l=[np.transpose(np.matrix([A[i,:p_na]])) for i in range(p_na)]
        el_right_r=[(np.kron(identity,np.kron(i,identity))) for i in L_i_r]
        el_left_r=[(np.kron(identity,np.kron(np.transpose(i),identity))) for i in L_i_r]
        el_right=[]
        el_left=[]
        for i in range(len(el_right_r)):
            for j in range(len(L_i_l)):
                el_right.append(np.kron(L_i_l[j],el_right_r[i]))
                el_left.append(np.kron(np.transpose(L_i_l[j]),el_left_r[i]))
    else: #|...,id1,...,id2,...>=|l_i3,id1,l_i2,id2,l_i1>=|l_i3,el1,el2>
        L_i_1=[np.transpose(np.matrix([A[i,:p_ne]])) for i in range(p_ne)]
        L_i_2=[np.transpose(np.matrix([A[i,:p_nc]])) for i in range(p_nc)]
        L_i_3=[np.transpose(np.matrix([A[i,:p_na]])) for i in range(p_na)]
        el_right_1=[np.kron(identity,i) for i in L_i_1]
        el_left_1=[np.kron(identity,np.transpose(i)) for i in L_i_1]
        el_right_2=[np.kron(identity,i) for i in L_i_2]
        el_left_2=[np.kron(identity,np.transpose(i)) for i in L_i_2]
        el_right=[]
        el_left=[]
        #|l_i3,el_right_2,el_right_1>
        for i in range(len(el_right_1)):
            for j in range(len(el_right_2)):
                for k in range(len(L_i_3)):
                    el_right.append(np.kron(L_i_3[k],np.kron(el_right_2[j],el_right_1[i])))
                    el_left.append(np.kron(np.transpose(L_i_3[k]),np.kron(el_left_2[j],el_left_1[i])))

    L_res=[]
    for j in range(len(D)):
        res=np.matrix(np.zeros((4,4)))
        for i in range(len(el_right)):
            res=res+np.matmul(el_left[i],np.matmul(D[j],el_right[i]))
        L_res.append(res)
    print(np.shape(L_res))
    tpsend=time.perf_counter()

    print("done Tr_",id1,",",id2," in :",tpsend-tpstart,"s")
    return L_res

def partial_trace_1_element(L_phi,id_element):#id element is the neutrino index which we will compute the partial trace #id element in [1,n]
    """This function compute the bipartite partial trace on one elements (end up with a 2*2 matrix).

    Args:
        L_phi (list of vector): list of the wavefunction at different time points
        id_element (int): id of the neutrino to trace on

    Returns:
        L_res (list of 2*2 matrix): Partial trace of the density matrix on one element id_element at different time points
    """

    tpstart=time.perf_counter()
    print("compute Tr_",id_element,":")
    D=C_density(L_phi)
    n=int(np.log(len(L_phi[0]))/np.log(2))#int(np.sqrt(len(L_phi[0])))#
    print(n)
    n_l=id_element-1#nb of element before the id_element
    n_r=n-id_element#nb of element after the id_element

    p_nr=2**n_r
    p_nl=2**n_l
    identity=np.matrix(np.eye(2))

    A=np.eye(2**(n-1))#matrice of all |i>x|j> exept id_element

    #construct el_left and el_right s.t.: tr_id_element(D)=Sum_i (el_left[i] *D* el_right[i])
    if n_l==0: #|. . . id_element> id_element is the last element of the vector
        L_i=[np.transpose(np.matrix([A[i,:]])) for i in range(2**(n-1))]
        el_right=[np.kron(i,identity) for i in L_i]
        el_left=[np.kron(np.transpose(i),identity) for i in L_i]
    elif n_r==0:#|id_element . . .> id_element is the first element of the vector
        L_i=[np.transpose(np.matrix([A[i,:]])) for i in range(2**(n-1))]
        el_right=[np.kron(identity,i) for i in L_i]
        el_left=[np.kron(identity,np.transpose(i)) for i in L_i]
    else: #|. . id_element . .> is in the bulk of the vector
        L_i_r=[np.transpose(np.matrix([A[i,:p_nr]])) for i in range(p_nr)]
        L_i_l=[np.transpose(np.matrix([A[i,:p_nl]])) for i in range(p_nl)]
        el_right_r=[np.kron(identity,i) for i in L_i_r]
        el_left_r=[np.kron(identity,np.transpose(i)) for i in L_i_r]
        el_right=[]
        el_left=[]
        for i in range(len(el_right_r)):
            for j in range(len(L_i_l)):
                el_right.append(np.kron(L_i_l[j],el_right_r[i]))
                el_left.append(np.kron(np.transpose(L_i_l[j]),el_left_r[i]))

    print(np.shape(el_right))
    print(np.shape(el_left))
    print(np.shape(D))
    # tr_id_element(D)=Sum_i (el_left[i] *D* el_right[i]) for every D(t)
    L_res=[]
    for j in range(len(D)):
        res=np.matrix(np.zeros((2,2)))
        for i in range(len(el_right)):
            res=res+np.matmul(el_left[i],np.matmul(D[j],el_right[i]))
        L_res.append(res)
    print(np.shape(L_res))
    tpsend=time.perf_counter()

    print("done Tr_",id_element," in :",tpsend-tpstart,"s")
    return L_res

def entropy(L_partial_trace):
    """This function compute the entropy of a single neutrino at all time.

    Args:
        L_partial_trace (list of 2*2 matrix): result of the partial trace on one element given by the function partial_trace_1_element

    Returns:
        out (list of float): Entropy on a single neutrino at different time points
    """
    out=[-np.trace(np.matmul(i,logm(i))/np.log(2)) for i in L_partial_trace]
    return out

def C_all_entropy(L_phi):
    """This function make the calculation of the entropy of all the neutrino at all the time points. it call the "entropy" function.

    Args:
        L_phi (list of vector): list of the wavefunction at different time points

    Returns:
        L_entropy (list list of float): List of list array like where we store the entropy of each neutrino at each time. L_entropy[i][idt] give the entropy of the neutrino i at time t.
    """
    n=int(np.log(len(L_phi[0]))/np.log(2))
    D=C_density(L_phi)
    L_entropy=[]
    for i in range(n):
        L_entropy.append(entropy(partial_trace_1_element(L_phi,i+1)))
    print(np.shape(L_entropy))
    return L_entropy

def C_all_entropy_1_and_2(L_phi):
    """This function compute the two body entropy for all the neutrino couple and at all time points. It also compute the one body entropy. At the end we also compute the fluctuation of the entropy.

    Args:
        L_phi (list of vector): the wavefunction at different time steps

    Returns:
        Mij (list storing the fluctuation coefficient of the entropy corresponding to neutrino i and j. Mij[int(Mij_id[i,j]] give the value of the fluctuation for neutrino i,j
        Mij_id (array of shape (n,n)): -1 if the id is not defined (because only the triangulare superior result are usefull) else it contain the id to the Mij element in the "Mij" list
        L_entropy_1 (list list of float): List of list array like where we store the entropy of each neutrino at each time. L_entropy_1[i][idt] give the entropy of the neutrino i at time t.
        L_entropy2 (list of list of float): List of list that store the entropy of each neutrino at each time. We need the "L_entropy_2_id" array to acces it's value. L_entropy2[int(L_entropy_2_id[i,j])][t] give the two body entropy for i and j neutrino at time t
        L_entropy_2_id (array of shape (n,n) of int): -1 if the id is not defined (because only the triangulare superior result are usefull) else it contain the id to the L_entropy2 element in the "L_entropy2" list
    """
    n=int(np.log(len(L_phi[0]))/np.log(2))
    D=C_density(L_phi)
    L_entropy_1=[]
    for i in range(n):
        L_entropy_1.append(entropy(partial_trace_1_element(L_phi,i+1)))
    print(np.shape(L_entropy_1))

    #-1 are discard value for symetrie reason we will only fill the higher diagonal of the matrix, the diagonal is not phyiscal so we will also discard it
    L_entropy_2_id=np.zeros((n,n))-1#contain the index in L_entropy2  of the element (i,j)
    L_entropy2=[]
    count=0
    for i in range(n):
        for j in range(i+1,n):
            L_entropy_2_id[i,j]=int(count)
            count+=1
            L_entropy2.append(entropy(partial_trace_2_element(L_phi,i+1,j+1)))#because of indexing

    Mij=[]
    Mij_id=np.zeros((n,n))-1
    count=0
    for i in range(n):
        for j in range(i+1,n):
            Mij_id[i,j]=int(count)
            count+=1
            L=[]
            for k in range(len(L_entropy2[0])):
                L.append(L_entropy_1[i][k]+L_entropy_1[j][k]-L_entropy2[int(L_entropy_2_id[i,j])][k])
            Mij.append(L)

    #L_entropy2[int(L_entropy_2_id[i,j])] is the function to get the value of the 2 body entropy in i,j for all neutrino
    #Mij[int(Mij_id[i,j])]
    return Mij,Mij_id,L_entropy_1,L_entropy2,L_entropy_2_id

##State preparation
#turn the state into the correct vector
def state_to_phi(vect):
    """This function take the initial state in flavor state and convert it to the corresponding wavefunction.

    Args:
        vect (list of int): Initial state in flavor state (ex: [0,1,1,0] is one for four neutrino)

    Returns:
        phi (vector): initial wavefunction corresponding to the initial flavor state (the dimension of it 2**n with n the number of neutrino/len of the initial flavor state)
    """
    tps1=time.perf_counter()
    phi=np.zeros(2**(len(vect)))
    def binary_to_number(l):
        num = 0
        for b in l:
            num = 2 * num + b
        return num
    phi[binary_to_number(vect)]=1
    tps2=time.perf_counter()
    print("phi0 is done in:",tps2-tps1,"s")
    return phi

##Save and load function old format


def save_Zi_l_t(L_t,Zi_t,path):
    """Fonction the save the Z operator measurement in the old format. (save the time points associated to it in a different file)

    Warning:
        the file are saved as arrays

    Args:
        L_t (list of float): list of the time points
        Zi_t (list list float): The value of each Z_i for each time point. Zi_t[i][t] give the value of the of the operator Z_i at the time t
        path (str): path at which we save the data

    Returns:
        void
        save the file corresponding to the time points and the array of the operator Z
    """
    print("save Zi_t")
    #print(Zi_t)
    print(np.shape(Zi_t))
    dim=len(Zi_t)
    print(dim)
    arr=np.array(Zi_t)
    with open(path+"saved_Zit_dim_"+str(dim)+".npy", 'wb') as f:
        np.save(f, arr)
    with open(path+"saved_tz_dim_"+str(dim)+".npy", 'wb') as f:
        np.save(f, L_t)
    print("Z_i is saved")

def load_Zi_l_t(dim,path):
    """Fonction the loas the Z operator measurement in the old format. (load the time points associated to it in a different file)

    Warning:
        the file are saved as arrays

    Args:
        dim (int): number of neutrino we are working with
        path (str): path at which we load the data

    Returns:
        L_t (list of float): list of the time points
        Zi_t (list list float): The value of each Z_i for each time point. Zi_t[i][t] give the value of the of the operator Z_i at the time t
    """
    print("loading Zi_t theoric classical solution:")
    with open(path+"saved_Zit_dim_"+str(dim)+".npy", 'rb') as f:
        Zi_t=np.load(f)
    with open(path+"saved_tz_dim_"+str(dim)+".npy", 'rb') as f:
        L_t=np.load(f)
    print("loading Zi_t theoric classical solution done")
    return L_t,Zi_t

def save_phi(dim,l_phi,ta,path,methode,ex_time):#ex_time=[th,tm,tz] th hamiltonian time, tm time of the methode used and tz time of the calcul of the operator
    """The function save the wavefunction, as well as the time points as well as the execution time of each part of the calculation

    Args:
        dim (int): number of neutrino we are working with
        l_phi (list of vector): list of the wavefunction at different time points
        ta (list of float): list of the time point used in the calculation
        path (str): path at which we save the data
        methode (int): id of the methode used to do the calculation
        ex_timr (list of len (3) of float): list containing the three element different time. The first one is the time to do the calculation of the Hamiltonian. The second one is the time to execute the methode choose and the third one is the time to compute the Z operator

    Returns:
        void
        save three file, one containing the wavefunction, the second containing the associated time points and a third one containing the calculation time.
    """
    arr=l_phi[0]
    for i in range(1,len(l_phi)):
        arr=np.concatenate((arr,l_phi[i]))
    print("shape arr",np.shape(arr))

    print(len(l_phi))
    print(np.shape(l_phi[0]))
    with open(path+"saved_phi_dim_"+str(dim)+"_nb_methode_"+str(methode)+".npy", 'wb') as f:
        np.save(f, arr)
    with open(path+"saved_t_dim_"+str(dim)+"_nb_methode_"+str(methode)+".npy", 'wb') as f:
        np.save(f, ta)
    with open(path+"saved_ex_time_dim_"+str(dim)+"_nb_methode_"+str(methode)+".npy", 'wb') as f:
        np.save(f, ex_time)


def load_phi(path,dim,methode):
    """The function load the wavefunction, as well as the time points as well as the execution time of each part of the calculation

    Args:
        path (str): path at which we save the data
        dim (int): number of neutrino we are working with
        methode (int): id of the methode used to do the calculation

    Returns:
        l_phi (list of vector): list of the wavefunction at different time points
        ta (list of float): list of the time point used in the calculation
        ex_timr (list of len (3) of float): list containing the three element different time. The first one is the time to do the calculation of the Hamiltonian. The second one is the time to execute the methode choose and the third one is the time to compute the Z operator
    """
    with open(path+"saved_phi_dim_"+str(dim)+"_nb_methode_"+str(methode)+".npy", 'rb') as f:
        arr=np.load(f)
    with open(path+"saved_t_dim_"+str(dim)+"_nb_methode_"+str(methode)+".npy", 'rb') as f:
        ta=np.load(f)
    with open(path+"saved_ex_time_dim_"+str(dim)+"_nb_methode_"+str(methode)+".npy", 'rb') as f:
        ex_time=np.load(f)
    n=len(ta)

    l_phi=[]
    step=2**dim
    for i in range(n):
        l_phi.append(np.transpose(np.matrix(arr[step*i:step*i+step,0])))
    return l_phi,ta,ex_time

##plot function
#plot the population Z in function of time
def plot_Z(ti,Zi):
    """This function plot the evolution of the different Zi operator with time

    Args:
        ti (list of float): list of the time points
        Zi (list list float): The value of each Z_i for each time point. Zi[i][t] give the value of the of the operator Z_i at the time t
    """
    for i in range(len(Zi)):
        plt.plot(ti,Zi[i],label=str(i))
    plt.xlabel("time: t",fontsize=18)
    plt.ylabel("<Z>_i(t)",fontsize=18)
    plt.legend(bbox_to_anchor=(0.8,0.70),fontsize=16)
    plt.show()

def plot_figure1():
    """this function generate a plot with three figure comparing the impact of each part of the Hamiltonian on the propagation. It does this propagation with it's own set's of parameter that we can modify by hand. They are define similarly to the "Application" section.

    Returns:
        void
        plot three different figure on a single plot. The top one correspond to the one body part only propagating with time (this correspond to the neutrino oscillation in the case of the mixing of state and given the two flavor approximation). The middle one correspond to the two body interraction only. The last one correspond to the propagation for the whole Hamiltonian. (save figure as the one generated by "plot_Z")
    """

    n_dim=4
    State=np.array([0,0,1,1])

    print("the initial state is:",State)
    phi0=np.transpose(np.matrix(state_to_phi(State)))#it's a vector
    print("phi0 is:",phi0)

    hbar=1 #hbar=1
    theta_nu=0.195#theta Amitrano
    C=1/n_dim #value used by alexandre
    b=C*np.array([np.sin(2*theta_nu),0,-np.cos(2*theta_nu)])
    J=Jij(n_dim,C)

    start=0
    finish= 200#200*10**(-6)
    step=1000
    ti=[(start+i*(finish-start)/step) for i in range(0,step)]#computation range

    L_Xi,L_Yi,L_Zi=XYZ_i(n_dim)
    H_1b=C_H_nu(L_Xi,L_Yi,L_Zi,b)
    H_2b=C_H_nunu(L_Xi,L_Yi,L_Zi,J)
    H_tot,L_Xi,L_Yi,L_Zi=C_H_n(n_dim,J,b)

    #methode exponential
    phi_t_1b=C_l_phi_t(H_1b,hbar,ti,phi0)
    phi_t_2b=C_l_phi_t(H_2b,hbar,ti,phi0)
    phi_t_tot=C_l_phi_t(H_tot,hbar,ti,phi0)

    #compute Z polulation
    Zi_1b=evaluate_operator_Z(l_Z,phi_t_1b)
    Zi_2b=evaluate_operator_Z(l_Z,phi_t_2b)
    Zi_tot=evaluate_operator_Z(l_Z,phi_t_tot)



    l_color=["b-","k--","g-.","r:"]

    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0.05)

    for i in range(len(Zi_1b)):
        axs[0].plot(ti,Zi_1b[i],l_color[i],label="neutrino "+str(i+1))
        axs[0].legend(loc="center right")


    for i in range(len(Zi_2b)):
        axs[1].plot(ti,Zi_2b[i],l_color[i])



    for i in range(len(Zi_tot)):
        axs[2].plot(ti,Zi_tot[i],l_color[i])

    axs[0].tick_params(axis="x", labelsize=14)
    axs[0].tick_params(axis="y", labelsize=14)
    axs[1].tick_params(axis="x", labelsize=14)
    axs[1].tick_params(axis="y", labelsize=14)
    axs[2].tick_params(axis="x", labelsize=14)
    axs[2].tick_params(axis="y", labelsize=14)

    axs[2].set_xlabel("Time $[µ^{-1}]$", fontsize=18)
    axs[0].set_ylabel("$<Z_{H_1}>(t)$", fontsize=18)
    axs[1].set_ylabel("$<Z_{H_2}>(t)$", fontsize=18)
    axs[2].set_ylabel("$<Z_{H_{tot}}>(t)$", fontsize=18)

    axs[0].text( 175, 0.50, '(a)',fontsize=18)
    axs[1].text( 175,0.55, '(b)',fontsize=18)
    axs[2].text( 175, 0.55, '(c)',fontsize=18)

    axs[0].set_xlim(0, 200)
    axs[1].set_xlim(0, 200)
    axs[2].set_xlim(0, 200)

    plt.show()


def plot_figure2():
    """This fonction load the waverfunction for 2,4,6,8 and 10 neutrinos and compute the Z operator before plothing them on the same plot but different figure. The parameter are define similarly to the "Application" section.

    We have introduce a few more coefficients:
        l_dim (l int): list of the dimension we want to compare. We could actually do the calculation for maybe 12 neutrino
    """
    l_dim=[2,4,6,8,10]
    #l_dim=[2,3,4,5,6,7,8]
    methode=1
    path_saved_file="/Users/bauge/Desktop/stage RP/recherche/code/"
    L_l_phi=[]
    L_ta=[]
    L_ex_time=[]
    L_l_Zi=[]

    l_style=["-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":"]#up to len(l_dim)=20
    l_letter=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t"]#up to len(l_dim)=20
    print("loading file")
    for i in range(len(l_dim)):
        L_Xi,L_Yi,L_Zi=XYZ_i(l_dim[i])
        l_phi,ta,ex_time=load_phi(path_saved_file,l_dim[i],methode)
        Zi=evaluate_operator_Z(L_Zi,l_phi)
        L_l_phi.append(l_phi)
        L_ta.append(ta)
        L_ex_time.append(ex_time)
        L_l_Zi.append(Zi)
    print("loading done")
    n=len(l_dim)


    fig, axs = plt.subplots(n, 1, figsize=(6, 8), sharex=True)
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0.05)
    for i in range(0,n):
        for j in range(l_dim[i]):
            print(len(L_ta[i]),len(L_l_Zi[i][j]))
            axs[i].plot(L_ta[i],L_l_Zi[i][j],linestyle =l_style[j],label="n"+str(j+1))
        axs[i].tick_params(axis="x", labelsize=14)
        axs[i].tick_params(axis="y", labelsize=14)
        axs[i].set_ylabel("$<Z>$n="+str(l_dim[i]), fontsize=18)
        axs[i].text( 175, 0.50, ("("+str(l_letter[i])+")"),fontsize=18)
        axs[i].set_xlim(0, 200)
    axs[n-1].legend(bbox_to_anchor=(1.1, (n/2)+0.75))
    axs[n-1].set_xlabel("Time $[µ^{-1}]$", fontsize=18)
    plt.show()

def plot_figure3(path_saved_file):
    """This function plot three figure on the same plot. It show the evolution of the time necessary to do each part of the calculation in function of the number of neutrino.
    """
    l_dim=[2,3,4,5,6,7,8,9,10]#[2,4,6,8]
    methode=1
    L_ex_time=[]

    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)


    print("loading file")
    for i in range(len(l_dim)):
        l_phi,ta,ex_time=load_phi(path_saved_file,l_dim[i],methode)
        L_ex_time.append(ex_time)
    l_th=[L_ex_time[i][0] for i in range(len(L_ex_time))]
    l_tm=[L_ex_time[i][1] for i in range(len(L_ex_time))]
    l_tz=[L_ex_time[i][2] for i in range(len(L_ex_time))]

    axs[0].plot(l_dim,l_th,marker='x')
    axs[1].plot(l_dim,l_tm,marker='x')
    axs[2].plot(l_dim,l_tz,marker='x')

    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')
    axs[0].set_ylabel("Hamiltonian $t[s]$", fontsize=18)
    axs[1].set_ylabel("Methode $t[s]$", fontsize=18)
    axs[2].set_ylabel("Operator $t[s]$", fontsize=18)

    axs[2].set_xlabel("number of neutrino", fontsize=18)

    plt.show()

def plot_s1s2s12_n4(path_saved_file):
    """This function load the wavefunction and compute the one body and two body entropy as well as the entropy fluctuation and compare on two different plot the case of the neutrino (1,2) and the case of the neutrino (1,4).
    """
    n=4
    methode=1
    l_phi,ta,ex_time=load_phi(path_saved_file,n,methode)
    l_style=["-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":"]
    Mij,Mij_id,L_entropy_1,L_entropy2,L_entropy_2_id=C_all_entropy_1_and_2(l_phi)

    fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    axs[0].plot(ti,L_entropy2[int(L_entropy_2_id[0,1])],linestyle =l_style[0],label="S12")
    axs[0].plot(ti,L_entropy_1[0],linestyle =l_style[1],label="S1")
    axs[0].plot(ti,L_entropy_1[1],linestyle =l_style[2],label="S2")
    axs[0].plot(ti,Mij[int(Mij_id[0,1])],linestyle =l_style[3],label="M12")
    axs[0].tick_params(axis="x", labelsize=14)
    axs[0].tick_params(axis="y", labelsize=14)
    axs[0].text( 175, 0.50, ("(a)"),fontsize=18)
    axs[0].set_xlim(0, 200)
    axs[0].legend()

    axs[1].plot(ti,L_entropy2[int(L_entropy_2_id[0,3])],linestyle =l_style[0],label="S14")
    axs[1].plot(ti,L_entropy_1[0],linestyle =l_style[1],label="S1")
    axs[1].plot(ti,L_entropy_1[3],linestyle =l_style[2],label="S4")
    axs[1].plot(ti,Mij[int(Mij_id[0,3])],linestyle =l_style[3],label="M14")
    axs[1].tick_params(axis="x", labelsize=14)
    axs[1].tick_params(axis="y", labelsize=14)
    axs[1].text( 175, 0.50, ("(b)"),fontsize=18)
    axs[1].set_xlim(0, 200)
    axs[1].set_xlabel("Time $[µ^{-1}]$", fontsize=18)
    axs[1].legend()

    plt.show()

def plot_2_body_average(path_saved_file,n=4):
    """This function load the wavefunction an compute use the "C_all_entropy_1_and_2" to get the two body entropy. From that it compute it's average over all couple of neutrino. It also plot the average for neutrino with the same initial state and with different initial state.
    """
    methode=1
    l_phi,ta,ex_time=load_phi(path_saved_file,n,methode)
    l_style=["-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":"]
    Mij,Mij_id,L_entropy_1,L_entropy2,L_entropy_2_id=C_all_entropy_1_and_2(l_phi)

    M_average=[]
    for j in range(len(Mij[0])):
        s=0
        for i in range(len(Mij)):
            s+=Mij[i][j]
        M_average.append(s/len(Mij))


    #compute l_state_changing_state and l_state_same_state with initial state for n even as |0,0,...,1,1>
    l_all=[]
    for i in range(n):
        for j in range(i+1,n):
            l_all.append([i,j])

    def test_s_or_c(n,a):
        #n is even and a=[i,j]
        diff=int(n/2)
        if a[0]<diff:
            if a[1]<diff:
                return 1
            else:
                return 0
        elif a[1]>diff:
            return 1
        else:
            return 0

    l_state_same_state=[]
    l_state_changing_state=[]

    for i in l_all:
        if test_s_or_c(n,i):
            l_state_same_state.append(i)
        else:
            l_state_changing_state.append(i)

    M_average_changing_state=[]
    #l_state_changing_state=[[0,2],[0,3],[1,2],[1,3]]
    for j in range(len(Mij[0])):
        s=0
        for i in range(len(l_state_changing_state)):
            s+=Mij[int(Mij_id[l_state_changing_state[i][0],l_state_changing_state[i][1]])][j]
        M_average_changing_state.append(s/len(l_state_changing_state))

    M_average_same_state=[]
    #l_state_same_state=[[0,1],[2,3]]
    for j in range(len(Mij[0])):
        s=0
        for i in range(len(l_state_same_state)):
            s+=Mij[int(Mij_id[l_state_same_state[i][0],l_state_same_state[i][1]])][j]
        M_average_same_state.append(s/len(l_state_same_state))

    plt.plot(ta,M_average,linestyle =l_style[0],label="M_average_all ")
    plt.plot(ta,M_average_changing_state,linestyle =l_style[1],label="M_average_changing_state ")
    plt.plot(ta,M_average_same_state,linestyle =l_style[2],label="M_average_same_state ")
    plt.legend()
    plt.show()


def plot_2_body_entropy_Mij(path_saved_file,n=4):
    """This function plot the two body entropy for every combinaison of neutrino. Their might be something wrong with this one
    """
    methode=1
    l_phi,ta,ex_time=load_phi(path_saved_file,n,methode)
    l_style=["-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":"]
    Mij,Mij_id,L_entropy_1,L_entropy2,L_entropy_2_id=C_all_entropy_1_and_2(l_phi)

    for i in range(0,n):
        for j in range(i+1,n):
            plt.plot(ta,Mij[int(Mij_id[i,j])],linestyle =l_style[i],label="neutrino "+str(i+1)+","+str(j+1))

    #for i in range(n):
    #    plt.plot(ta,all_entropy[i],linestyle =l_style[i],label="neutrino "+str(i+1))
    plt.legend()
    plt.show()

def plot_entropy(path_saved_file,n=4):
    """This function load the wavefunction and compute the one neutrino entropy associated to it. Then it plot it.
    """
    methode=1
    l_phi,ta,ex_time=load_phi(path_saved_file,n,methode)
    l_style=["-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":"]
    all_entropy=C_all_entropy(l_phi)
    for i in range(n):
        plt.plot(ta,all_entropy[i],linestyle =l_style[i],label="neutrino "+str(i+1))
    plt.legend()
    plt.show()

def plot_all_entropy(path_saved_file):
    """This function load the wavefunction for all the dimension in l_dim and compute the one body entropy. Then it plot all the result in different figure on the same plot. This function can take more than 8Go of memory given these parameter.

    l_dim (list of int): list of dimension we want to plot the entropy of
    """

    l_dim=[2,3,4,5,6,7,8]
    #l_dim=[2,3,4,5,6,7,8]
    methode=1
    L_l_entropy=[]
    L_ta=[]


    l_style=["-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":"]#up to len(l_dim)=20
    l_letter=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t"]#up to len(l_dim)=20
    print("loading file")
    for i in range(len(l_dim)):
        l_phi,ta,ex_time=load_phi(path_saved_file,l_dim[i],methode)
        L_l_entropy.append(C_all_entropy(l_phi))
        L_ta.append(ta)
    print("loading done")
    n=len(l_dim)

    fig, axs = plt.subplots(n, 1, figsize=(6, 8), sharex=True)
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0.05)
    for i in range(0,n):
        for j in range(l_dim[i]):
            axs[i].plot(L_ta[i],L_l_entropy[i][j],linestyle =l_style[j],label="n"+str(j+1))
        axs[i].tick_params(axis="x", labelsize=14)
        axs[i].tick_params(axis="y", labelsize=14)
        axs[i].set_ylabel("S n="+str(l_dim[i]), fontsize=18)
        axs[i].text( 180, 0.25, ("("+str(l_letter[i])+")"),fontsize=18)
        axs[i].set_xlim(0, 200)
    axs[n-1].legend(bbox_to_anchor=(1.1, (n/2)+0.75))
    axs[n-1].set_xlabel("Time $[µ^{-1}]$", fontsize=18)
    plt.show()

def plot_1b_2b_entropy(n=4):
    """This function compute the entropy related to only the one body part of the Hamiltonian, then only the two body part then the whole Hamiltonian. It finally plot everything on the same plot but different figure, from top to bottom. You have to set the parameter independently of "Application" part but the same way. It's not loading or saving any data.
    """
    n_dim=n
    if n_dim==2:
        State=np.array([0,1])#in 2D
    elif n_dim==3:
        State=np.array([0,0,1])
    elif n_dim==4:
        State=np.array([0,0,1,1])#in 4D: initial state of Amitrano for N=4 (we have to turn it into the correct size of vector first
    elif n_dim==5:
        State=np.array([0,0,0,1,1])
    elif n_dim==6:
        State=np.array([0,0,0,1,1,1])
    elif n_dim==7:
        State=np.array([0,0,0,0,1,1,1])
    elif n_dim==8:
        State=np.array([0,0,0,0,1,1,1,1])
    elif n_dim==9:
        State=np.array([0,0,0,0,0,1,1,1,1])
    elif n_dim==10:
        State=np.array([0,0,0,0,0,1,1,1,1,1])
    elif n_dim==11:
        State=np.array([0,0,0,0,0,0,1,1,1,1,1])

    hbar=1 #hbar=1
    theta_nu=0.195#theta Amitrano
    C=1/n_dim #value used by alexandre
    b=C*np.array([np.sin(2*theta_nu),0,-np.cos(2*theta_nu)])
    J=Jij(n_dim,C)

    phi0=np.transpose(np.matrix(state_to_phi(State)))

    start=0
    finish= 200#200*10**(-6)
    step=1000
    ti=[(start+i*(finish-start)/step) for i in range(0,step)]#computation range

    L_Xi,L_Yi,L_Zi=XYZ_i(n_dim)
    H_1b=C_H_nu(L_Xi,L_Yi,L_Zi,b)
    H_2b=C_H_nunu(L_Xi,L_Yi,L_Zi,J)
    H_tot,L_Xi,L_Yi,L_Zi=C_H_n(n_dim,J,b)

    print(np.shape(phi0))

    phi_t_1b=C_l_phi_t(H_1b,hbar,ti,phi0)
    phi_t_2b=C_l_phi_t(H_2b,hbar,ti,phi0)
    phi_t_tot=C_l_phi_t(H_tot,hbar,ti,phi0)

    l_entropy_1b=C_all_entropy(phi_t_1b)
    l_entropy_2b=C_all_entropy(phi_t_2b)
    l_entropy_al=C_all_entropy(phi_t_tot)

    L=[l_entropy_1b,l_entropy_2b,l_entropy_al]
    L_y_label=["S 1b","S 2b", "S"]
    l_style=["-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":"]#up to len(l_dim)=20
    l_letter=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t"]#up to len(l_dim)=20

    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    for i in range(len(L)):
        for j in range(len(L[i])):
            axs[i].plot(ti,L[i][j],linestyle =l_style[j],label="n"+str(j+1))
        axs[i].tick_params(axis="x", labelsize=14)
        axs[i].tick_params(axis="y", labelsize=14)
        axs[i].set_ylabel(L_y_label[i], fontsize=18)
        axs[i].text( 175, 0.50, ("("+str(l_letter[i])+")"),fontsize=18)
        axs[i].set_xlim(0, 200)
    axs[2].legend(bbox_to_anchor=(1.1, (n/2)))
    axs[2].set_xlabel("Time $[µ^{-1}]$", fontsize=18)
    plt.show()

def plot_entropy_average(path_saved_file):
    """This fonction load the wavefunction and compute the average entropy for some number of neutrino. It them plot every in the same figure. This function also need more that 9Go of memory to use do the number of point use.
    """
    #l_dim=[2,3,4]
    l_dim=[2,3,4,5,6,7,8]
    methode=1
    L_l_entropy=[]
    L_ta=[]

    l_style=["-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.",":"]#up to len(l_dim)=20
    l_letter=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t"]#up to len(l_dim)=20
    print("loading file")
    for i in range(len(l_dim)):
        l_phi,ta,ex_time=load_phi(path_saved_file,l_dim[i],methode)
        L_l_entropy.append(C_all_entropy(l_phi))
        L_ta.append(ta)
    print("loading done")

    n=len(l_dim)
    m=len(L_ta[0])
    print(n,m)
    len(L_l_entropy)
    L_average=[[np.sum(np.array(L_l_entropy[j])[:,i])/(len(L_l_entropy[j])) for i in range(m)]for j in range(n)]

    for i in range(len(L_average)):
        plt.plot(L_ta[0],L_average[i],linestyle =l_style[i],label="n="+str(l_dim[i]))
    plt.xlabel("Time $[µ^{-1}]$", fontsize=18)
    plt.ylabel("S", fontsize=18)
    plt.legend(bbox_to_anchor=(1.1, 1/2))
    plt.show()
    return L_l_entropy
##Data new format
def data_to_new_format(N,path_data_new_format,path_saved_file,do_entropy=False):

    path_saved_file="/Users/bauge/Desktop/stage RP/recherche/code/"
    """This function compute different quantity and put them in the same format as quantum computation before saving them at the path "path_data_new_format". It load the wavefunction from the path path_saved_file
    It compute the following variables:
        * entropy
        * Z operator
        * X operator
        * Y operator
        * ZZ operator
        * Czz, Cdzz, Codzz fluctuation terms
        * approximate entropy

    Args:
        N (int): the number of neutrino in the system
        path_data_new_format (str): path to where we will same the computation
        path_saved_file (str): path from where we load the wavefunction

    Returns:
        void
        save everything to "path_data_new_format" with the function "save_data_new_format".

    """
    methode=1
    if do_entropy:
         data_entropy_new_format(N,path_data_new_format,path_saved_file)

    step_id=5

    l_phi,ta,ex_time=load_phi(path_saved_file,N,methode)
    l_temps=ta[::step_id]

    l_phi=[l_phi[t] for t in range(0,len(l_phi),step_id)]


    #evaluate Z_i
    L_Xi,L_Yi,L_Zi=XYZ_i(N)
    Zi=np.array(evaluate_operator_Z(L_Zi,l_phi))
    Xi=np.array(evaluate_operator_Z(L_Xi,l_phi))
    Yi=np.array(evaluate_operator_Z(L_Yi,l_phi))

    #evaluate Zi_Zj
    tab_ZZ=evaluate_ZZ_operator(N,l_phi)

    print(np.shape(Zi))
    #print(Zi)
    print(np.shape(tab_ZZ))
    print(np.shape(l_temps))

    tab_Czz_Cdzz_codzz=compute_Czz_Cdzz_Codzz(Zi,tab_ZZ,l_temps)

    tab_approximate_average_one_entropy=compute_approximate_one_entropy(l_temps,Zi)

    Z_i_f=concatenate_t_Z(Zi,l_temps)
    X_i_f=concatenate_t_Z(Xi,l_temps)
    Y_i_f=concatenate_t_Z(Yi,l_temps)

    print(np.shape(Z_i_f))

    save_data_new_format(Z_i_f,tab_Czz_Cdzz_codzz,tab_approximate_average_one_entropy, path_data_new_format,X_i_f,Y_i_f)


def save_data_new_format(tab_t_Z,tab_Czz_Cdzz_Codzz_f,tab_approximate_entropy_f,path,X_i_f,Y_i_f):
    """This fonction save all the calculation done in the new format on a file.

    Args:
        tab_t_Z (array): tab_t_Z[idt,i+1] give the value of the Z operator for neutrino i at time the time step idt. Result are given in the new format. And tab_t_Z[idt,0] give the time corresponding at the time step idt
        tab_Czz_Cdzz_Codzz_f (array): tab_Czz_Cdzz_Codzz_f (array of shape (nbt,4)): Array that all the result. The first column tab_Czz_Cdzz_Codzz_f[:,0] store the time, the second column store tab_Czz_Cdzz_Codzz_f[:,1] store the czz measure, the third column tab_Czz_Cdzz_Codzz_f[:,2] store the cdzz, the third columns store the codzz.
        tab_approximate_entropy_f (array): Array where the first column correspond to the time point and the second collumns store the average approximate entropy. tab_approximate_entropy_f[idt,0] give the time corresponding to the time step idt. tab_approximate_entropy_f[idt,1] give the average approximate entropy at the time step idt.

        path (str): path to which we will save the data
        X_i_f (array): X_i_f[idt,i+1] give the value of the X operator for neutrino i at time the time step idt. Result are given in the new format. And X_i_f[idt,0] give the time corresponding at the time step idt. This is the result that is not trustworthy.
        Y_i_f (array): Y_i_f[idt,i+1] give the value of the Y operator for neutrino i at time the time step idt. Result are given in the new format. And Y_i_f[idt,0] give the time corresponding at the time step idt. This is the result that is not trustworthy.

    Returns:
        void
        Create multiple file to save the data to the file path given by "path"
    """
    n_temps,n_neutrino=np.shape(tab_t_Z)
    n_neutrino=n_neutrino-1
    print(n_neutrino)

    np.savetxt(path+"n="+str(n_neutrino)+"_classical_Zi_per_neutrino.dat",np.real(tab_t_Z))
    np.savetxt(path+"n="+str(n_neutrino)+"_classical_Xi_per_neutrino.dat",np.real(X_i_f))
    np.savetxt(path+"n="+str(n_neutrino)+"_classical_Yi_per_neutrino.dat",np.real(Y_i_f))
    np.savetxt(path+"n="+str(n_neutrino)+"_classical_Czz_Cdzz_Codzz.dat",tab_Czz_Cdzz_Codzz_f)

    np.savetxt(path+"n="+str(n_neutrino)+ "_classical_Entropy_average_approximate.dat",tab_approximate_entropy_f)

def concatenate_t_Z(tab_Z,l_temps):
    """This function is the function that change the format of data. It exange the place of the index i and t. Then it add one more collumns we place at the begining correspond to the time point. This function also work for tab_X and tab_Y.

    Args:
        tab_Z (array): Zi_t[i,t] give the value of the Z for neutrino i at time t. Result of the Z operator in the old format.
        l_temps (list of float): List of time point used in the calculation

    Returns:
        b (array): b[idt,i+1] give the value of the Z for neutrino i at time the time step idt. Result are given in the new format. And b[idt,0] give the time corresponding at the time step idt
    """
    a=np.array(list(zip(*tab_Z[::-1]))) #rotate the table by -pi/2 change collumns to ligne here Zn-1 Zn-2 ... Z0
    b=np.c_[l_temps,a[::,::-1]] #add the time and put the correct order of Zi, here temps, Z0 Z1 ... Zn
    return b

def compute_approximate_one_entropy(l_temps,tab_Z):
    """This function compute the approximate average one entropy using the Zi operators.

    Args:
        l_temps (list of float): list of the time points used in the calculation
        tab_Z (array): Zi_t[i,t] give the value of the Z for neutrino i at time t. Result of the Z operator in the old format.

    Returns:
        tab_out (array): Array where the first column correspond to the time point and the second collumns store the average approximate entropy. tab_out[idt,0] give the time corresponding to the time step idt. tab_out[idt,1] give the average approximate entropy at the time step idt.
    """
    n_neutrino,nb_temps=np.shape(tab_Z)
    tab_out=np.zeros((nb_temps,2))
    tab_out[:,0]=l_temps

    for t in range(nb_temps):
        out=0
        for i in range(n_neutrino):
            p0=(1/2)*(1+tab_Z[i,t])
            p1=(1/2)*(1-tab_Z[i,t])
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




def evaluate_ZZ_operator(N,l_phi):
    """This function evaluate the ZZ operator for all the couple of neutrino and all the time point.

    Args:
        N (int): the number of neutrino we are using
        l_phi (list of vector): list of the wavefunction at different time points.

    Returns:
        tab_out (array of shape (nb_qubit,nb_qubit,nbt): array storing the Zij(t) result. tab_out[i,j,idt] give the value of the Zij operator at the time step idt.
    """

    def ZZ_eval(Z,phi_t_eval):
        """This function measure the ZZ operator given the ZZij matrix and the list of wavefunction.

        Args:
            Z (matrix): matrix form of the operator
            phi_t_eval (vector): one wavefunction at a specific time points

        Returns:
            The value of the ZZ operator given the operator it act to and the time point from the wavefunction
        """
        right=np.matmul(Z,phi_t_eval)
        phi_t_H=phi_t_eval.getH()
        out=np.matmul(phi_t_H,right)
        return out[0,0]

    def construct_ZZ_operator(id_i,id_j,N):
        """This fonction compute the ZZ matrix corresponding to the neutrn id_i and id_j in the case of N neutrino in total.

        Args:
            id_i (int): id of the first neutrino
            id_j (int): id of the second neutrino
            N (int): number of neutrino

        Returns:
            s (matrix): matrix form of the ZiZj operator on N neutrino

        """
        Id=np.eye(2)
        SigmaZ=np.matrix([[1,0],[0,-1]])
        s=np.matrix([1])
        for i in range(N):
            if id_i==id_j:#SigmaZ**2=Id, ZiZi operator is identity
                s=np.kron(s,Id)

            #if id_i!= id_j:
            elif i==id_i:
                s=np.kron(s,SigmaZ)
            elif i==id_j:
                s=np.kron(s,SigmaZ)
            else:
                s=np.kron(s,Id)
        return s

    ndt,n_state_2,_=np.shape(l_phi)
    nb_qbit=int(np.sqrt(n_state_2))

    tab_out=np.zeros((nb_qbit,nb_qbit,ndt)) #tab[id_qbit,idt]
    for i in range(N):
        for j in range(N):
            ZZ_operator=construct_ZZ_operator(i,j,N)
            for t in range(ndt):
                #print(l_phi[t])
                a=np.matrix(l_phi[t])
                #print(np.shape(a))
                #print(ZZ_eval(ZZ_operator,a)[0,0])
                tab_out[i,j,t]=ZZ_eval(ZZ_operator,a)
    #print(tab_out[0,1,:])
    return tab_out


def data_entropy_new_format(N,path_data_new_format,path_saved_file):
    """Function that load the wavefunction from the "path_saved_file" and compute the entropy then the average entropy. It finally save the average entropy in a file at the path "path_data_new_format".

    Args:
        N (int): number of neutrino
        path_data_new_format (str): path to which we save the date
        path_saved_file (str): path to which we load the wavefunction

    Returns:
        void
        save the average entropy to the location "path_data_new_format"
    """

    methode=1

    l_phi,ta,ex_time=load_phi(path_saved_file,N,methode)
    L_entropy=C_all_entropy(l_phi)

    #if n_dt=1000,we want to keep 200 step, one every 5 step
    l_temps=ta[::5]

    print(l_temps)
    print(len(l_temps))

    L_entropy=np.array(L_entropy)[:,::5]
    nb_neutrino,nb_temps=np.shape(L_entropy)


    l_av_entropy=[]
    for t in range(len(L_entropy[0])):
        s=0
        for i in range(len(L_entropy)):
            s=s+L_entropy[i][t]
        l_av_entropy.append(s/(len(L_entropy)))

    tab_entropy_f=np.zeros((nb_temps,nb_neutrino+1))
    tab_entropy_f[:,0]=l_temps

    for i in range(nb_neutrino):
        tab_entropy_f[:,i+1]=L_entropy[i,:]
    print(tab_entropy_f)

    tab_av_entropy_f=np.zeros((nb_temps,2))
    tab_av_entropy_f[:,0]=l_temps
    tab_av_entropy_f[:,1]=l_av_entropy

    np.savetxt(path_data_new_format+"n="+str(nb_neutrino)+"_classical_Exact_average_entropy_per_neutrino.dat",tab_av_entropy_f)

def plot_1_body_oscillation(l_t,Z):
    """This function plot the one body oscillation, it must be used only in the case of n=1, else it will display the two body interraction too.
    """
    print(np.shape(Z))
    l_z=[1/2*(1-Z[0][i]) for i in range(len(Z[0]))]
    l_z2=[1-l_z[i] for i in range(len(l_z))]
    plt.plot(l_t,l_z,label="$nu_x$")
    plt.plot(l_t,l_z2,label="$nu_e$")
    plt.xlabel("time: t",fontsize=18)
    plt.ylabel("Probability",fontsize=18)
    plt.legend()
    plt.show()


##main
"""This section contain some already prepared code to directly use the function presented ealier.

Before executing the code, you need to check the path at which data will save and load from:
    - The "path_saved_file" is the location where we put both classical and quantum result in the old format (this is where we store the calculation we are loading back.
    - The "path_data_new_format" is the location of the new format of data.

In order to use them you can interract with the following parameter:
    - the initial state is controled by "phi0" right bellow
    - hbar value is set to 1
    - theta_nu is the mixing angle in the case of two flavor approximation
    - C is the main coefficient of the two body interraction
    - b is the main coefficient of the one body interraction
    - start give at which point we start the propagation from
    - finish give at which point we finish the propagation
    - step give how many step we will do to propagate from start to finish
    - ti list of time points
    - methode_select control which part of the code we will run

The methode_select parameter control which task you want to do:

*methode_select==1 Do the calculation of the wavefunction and the Z operator before saving everything in path_saved_file as methode 1. This is the main methode to do the calculation.

*methode_select==2 Do the calculation of the wavefunction and the Z operator but with the diagolnalisation methode (we don't save the result because we choose not to use it)

*methode_select==3 Do the calculation of the wavefunction and the Z operator but with the euler method (we don't save the result because we choose not to use it)

*methode_select==4 Do the calculation of the wavefunction and the Z operator but with the Runge-Kutta 2 methode (it save the result but we also ended up not using this methode, it saved the result as methode 4 and save them at path_saved_file)

*methode_select==5 Only do the calculation of the Hamiltonian (was used to choose the parameter of the QPE methode see the quantum code of the two flavor approximation)

*methode_select==6 Here are the call of most of the plot used in the report. Each line can be run separately.

*methode_select==7 This methode do the calculation of the Z operator, the entropy, the fluctuation and format everything to be usable as a reference for the quantum computation.It save everything at "path_data_new_format" in the new_format for the operator Z array. It's loading the result from methode_select==1 at the location "path_saved_file"

"""

methode_select=1#1 for brut exponential, 2 for base changes,3 for the most schrodinger equation and 4 for the optimised 3.
#select dim:
n_dim=4

th=-1 #runing time of H, of the methode and of Z
tm=-1
tz=-1
if n_dim==1:
    State=np.array([0])
if n_dim==2:
    State=np.array([0,1])#in 2D
elif n_dim==3:
    State=np.array([0,0,1])
elif n_dim==4:
    State=np.array([0,0,1,1])#in 4D: initial state of Amitrano for N=4 (we have to turn it into the correct size of vector first
elif n_dim==5:
    State=np.array([0,0,0,1,1])
elif n_dim==6:
    State=np.array([0,0,0,1,1,1])
elif n_dim==7:
    State=np.array([0,0,0,0,1,1,1])
elif n_dim==8:
    State=np.array([0,0,0,0,1,1,1,1])
elif n_dim==9:
    State=np.array([0,0,0,0,0,1,1,1,1])
elif n_dim==10:
    State=np.array([0,0,0,0,0,1,1,1,1,1])
elif n_dim==11:
    State=np.array([0,0,0,0,0,0,1,1,1,1,1])
elif n_dim==12:
    State=np.array([0,0,0,0,0,0,1,1,1,1,1,1])

print("the initial state is:",State)
phi0=np.transpose(np.matrix(state_to_phi(State)))#it's a vector
print("phi0 is:",phi0)

self.path_saved_file="./data_2_flavor/"
self.path_data_new_format="./data_2_flavor_formated_quantum/"

#-------------------------------------------------------------------
#parameter of the hamiltonian
hbar=1 #hbar=1
#Amitrano's parameter
#For the unkown coefficient alexandre choose to do:
#mu/N=delta(m)**2/4E (which is always correct) and choose the energy #scale mu=1
theta_nu=0.195#theta Amitrano
C=1/n_dim #value used by alexandre
b=C*np.array([np.sin(2*theta_nu),0,-np.cos(2*theta_nu)])
J=Jij(n_dim,C)

#-------------------------------------------------------------------
#plot and range parameter
start=0
finish= 200#200*10**(-6)
step=10000
ti=[(start+i*(finish-start)/step) for i in range(0,step)]#computation range
#-------------------------------------------------------------------
#compute H
tps1=time.perf_counter()
H_eval,l_X,l_Y,l_Z=C_H_n(n_dim,J,b)
tps2=time.perf_counter()
print("computing time of H:",tps2-tps1)
th=tps2-tps1
#print(H_eval)

#-------------------------------------------------------------------
print("methode:",methode_select)
if methode_select==1:
    #H_eval the hamiltonian
    #phi0 the initial wave function
    #ti

    print("compute U(t) for all t, then all phi(t)")
    tpsa=time.perf_counter()
    l_phi_t=C_l_phi_t(H_eval,hbar,ti,phi0)
    tpsb=time.perf_counter()
    tm=tpsb-tpsa
    print("we have all phit in: ",tm)

    save_phi(n_dim,l_phi_t,ti,path_saved_file,methode_select,[th,tm,0])

    tpsa=time.perf_counter()
    Zi=evaluate_operator_Z(l_Z,l_phi_t)
    tpsb=time.perf_counter()
    tz=tpsb-tpsa

    save_Zi_l_t(ti,Zi,path_saved_file)

    save_phi(n_dim,l_phi_t,ti,path_saved_file,methode_select,[th,tm,tz])
    load_Zi_l_t(n_dim,path_saved_file)

    print("ploting")
    plot_Z(ti,Zi)

    if n_dim==1:
        plot_1_body_oscillation(ti,Zi)

elif methode_select==2:
    #H_eval the hamiltonian
    #phi0 the initial wave function
    #ti
    #main methode 2
    #-------------------------------------------------------------------
    #compute and H and phi0 depending on constant
    eigenvalues,eigenvectors=diagonalise_H_m2(H_eval)

    l_phi_t=[cnt(phi0,eigenvalues,eigenvectors,i,hbar) for i in ti]

    #l_phi_t are the phi(t) in the initial basis
    #-------------------------------------------------------------------
    #plot on Z:
    Zi=evaluate_operator_Z(l_Z,l_phi_t)

    plot_Z(ti,Zi)

elif methode_select==3:#solve the differential equation using euler
    #this methode doesn't converge because euler introduce to much error, a rk2 is preferable
    #H_eval the hamiltonian
    #phi0 the initial wave function
    #ti

    print("Euler solution:")

    l_phi_t=propagate_time(phi0,ti,hbar,H_eval)

    print("Done computation")
    Zi=evaluate_operator_Z(l_Z,l_phi_t)

    print("ploting")
    print(len(Zi))

    plot_Z(ti,Zi)
elif methode_select==4:#methode 3 with range kutta
    #H_eval the hamiltonian
    #phi0 the initial wave function
    #ti

    print("Runge kutta 2:")
    tpsa=time.perf_counter()
    l_phi_t=cnt_w_RK2(phi0,ti,hbar,H_eval)
    tpsb=time.perf_counter()
    tm=tpsb-tpsa

    print("Done")

    tpsa=time.perf_counter()
    Zi=evaluate_operator_Z(l_Z,l_phi_t)
    tpsb=time.perf_counter()
    tz=tpsb-tpsa
    save_phi(n_dim,l_phi_t,ti,path_saved_file,methode_select,[th,tm,tz])
    print("ploting")
    plot_Z(ti,Zi)

elif methode_select==5:
    theta_nu=0.195#theta Amitrano
    C=1/n_dim #value used by alexandre
    b=C*np.array([np.sin(2*theta_nu),0,-np.cos(2*theta_nu)])

    n=8
    J=Jij(n,C)
    H,l_X,l_Y,l_Z=C_H_n(n,J,b)

elif methode_select==6:#rest of the plot function, then can be run totaly independently
    plot_figure1()
    plot_figure2()
    plot_figure3(path_saved_file)
    plot_s1s2s12_n4(path_saved_file)
    plot_2_body_average(path_saved_file,n=4)
    plot_2_body_entropy_Mij(path_saved_file,n=4)
    plot_entropy(path_saved_file,n=4)
    plot_all_entropy(path_saved_file)
    plot_1b_2b_entropy(n=4)
    plot_entropy_average(path_saved_file)

elif methode_select==7:#Compute the new format of dat
    data_to_new_format(N,path_data_new_format,path_saved_file,do_entropy=False)
    data_entropy_new_format(N,path_data_new_format,path_saved_file)