"""
Program som inneholder alle funksjoner for å gjøre alle simuleringene
"""
import numpy as np
from scipy.sparse import diags, eye, kron, csr_matrix
from scipy.sparse.linalg import factorized

def isNormalized(U: np.ndarray, tol: float = 1e-10) -> bool:
    return np.isclose(np.sum(np.abs(U)**2), 1.0, atol=tol)

def totalPOvertime(U: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(U)**2, axis=(1, 2))

def normalize(U: np.ndarray) -> np.ndarray:
    factor = np.sqrt(np.sum(np.abs(U)**2))
    return U/factor

def initialCondition(
        nH: int, 
        x_c: float = 0.25, y_c: float = 0.50,
        sigma_x: float = 0.05, sigma_y: float = 0.1,
        p_x: float = 200.0, p_y: float = 0.0,
        ):
    """
    Lager initialtilstanden til systemet som en Gaussisk Bølge.
    nH er antall punkter i x og y rettning. x_c og y_c er koordinatene til senteret av bølgepakken.
    sigma_x og sigma_y gir bredden i x og y rettning (standardavviket)
    p_x og p_y er bevegelsemengden i x og y rettning
    Funksjonen gir ut U0 som en nH x nH matrise og to meshgridder X og Y av samme dimensjon.
    """
    x = np.linspace(0,1,nH)
    y = np.linspace(0,1,nH)
    X, Y = np.meshgrid(x, y, indexing='ij')
    U0 = np.exp(-(X-x_c)**2/(2*sigma_x**2)-(Y-y_c)**2/(2*sigma_y**2)+1j*p_x*X+1j*p_y*Y)

    # Dirichlet på randen: u = 0 på alle kanter
    U0[0, :]  = 0
    U0[-1, :] = 0
    U0[:, 0]  = 0
    U0[:, -1] = 0

    U0 = normalize(U0)
    return U0, X, Y

def convertToP(U):
    return np.abs(U)**2

def v(X: np.ndarray, Y: np.ndarray, v0: float = 1e10) -> np.ndarray:
    """
    Tar inn X, Y (2d matriser fra meshgrid) og v0 som er hva potensiale skal være med veggen.
    Funksjonen lager en matrise som svarer til den diskrete potensialfunksjonen for
    dobbelspalte eksperimentet.
    """
    cond_x = (X >= 0.49) & (X <= 0.51)
    cond_y = (
        ((Y >= 0.475) & (Y <= 0.525)) |
        ((Y >= 0.0)   & (Y <= 0.425)) |
        ((Y >= 0.575) & (Y <= 1.0))
    )

    mask = cond_x & cond_y

    V = np.zeros_like(X, dtype=float)
    V[mask] = v0
    return V

def build_H(nH: int, V: np.ndarray) -> csr_matrix:
    """
    Tar inn nH, antall punkter langs x og y aksen for gridden
    Tar inn V, som er den diskre potensialfunksjonen som en 2d array
    Bygger hamilton operatoren som en diskre matrise av sparse.
    Er ment til å bli brukt på u som er gjort til en 1d array ved ravel()
    """
    def laplacian(n_in,h):
        main = -2.0 * np.ones(n_in)
        off  =  1.0 * np.ones(n_in-1)
        L =  diags([off, main, off], [-1, 0, 1]) # type: ignore
        return L/ h**2
    
    n_in = nH - 2

    h = 1.0 / (nH - 1)
    
    L = laplacian(n_in, h)
    I = eye(n_in, format="csr")
    L2 = kron(I, L) + kron(L, I)   # størrelse (nH*nH, nH*nH)

    V_inner = V[1:-1, 1:-1].ravel()
    Vmat = diags(V_inner, 0, format="csr") # gjør den til lang 1d array
    H = -0.5*L2+Vmat
    return H


def setup_time_evolution(u0: np.ndarray, nT: int,t_end: float, nSkip: int, max_memory_gb: float):
    """
    Vanlig setup for alle skjema
    returnerer (nH, n_in, nFrames, dt, t, U) hvr U har shape (nFrames+1, nH, nH).
    """
    nH = u0.shape[0]
    n_in = nH - 2
    nFrames = nT // nSkip

    dtype = u0.dtype
    bytes_per_item = np.dtype(dtype).itemsize
    mem_gb = ((nFrames + 1) * nH * nH * bytes_per_item) / (1024**3)
    if mem_gb > max_memory_gb:
        raise MemoryError(
            f"U trenger mere minne enn grensen tilgir. Trenger {mem_gb:.2f} GB"
        )

    U = np.zeros((nFrames + 1, nH, nH), dtype=dtype)

    U[0] = u0

    dt = t_end / (nT - 1)
    t = np.arange(nFrames + 1) * nSkip * dt

    return nH, n_in, nFrames, dt, t, U

def schrod_rhs(H: csr_matrix, u: np.ndarray, k: int | None = None):
    v = H @ u

    if not np.isfinite(u).all():
        raise ValueError(f"u contains NaN or Inf at iteration {k}")

    if not np.isfinite(v).all():
        raise ValueError(f"H @ u contains NaN or Inf at iteration {k}")

    return v

def rk2_step(u,H,dt):
    k1 = -1j * (H @ u)
    k2 = -1j * (H @ (u + 0.5 * dt * k1))
    return u + dt * k2

def leapfrog_step(u,u_prev,H,dt,k):
    v = schrod_rhs(H,u,k)
    return u_prev - 2j * dt * v

def rk4_step(u,H,dt, k):
    v = schrod_rhs(H,u,k)
    k1 = -1j * (H@u)
    k2 = -1j * (H @ (u + 0.5 * dt * k1))
    k3 = -1j * (H @ (u + 0.5 * dt * k2))
    k4 = -1j * (H @ (u + dt * k3))
    return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def finite_diff_step(u,H,dt,k):
    v = schrod_rhs(H,u,k)
    return u - 1j * dt * v


def finite_diff(u0: np.ndarray, H: csr_matrix, nT: int, t_end: float, nSkip: int = 1, max_memory_gb: float = 1.0):
    nH, n_in, nFrames, dt, t, U = setup_time_evolution(u0, nT, t_end, nSkip, max_memory_gb)
    u = U[0, 1:-1, 1:-1].ravel()
    global_step = 1
    for k in range(1,nFrames+1):
        for _ in range(nSkip):
            u = finite_diff_step(u, H, dt,global_step)
            global_step += 1
        U_k = np.zeros((nH, nH), dtype=U.dtype)
        U_k[1:-1, 1:-1] = u.reshape(n_in, n_in)
        U[k] = U_k
    return t, U

def leapfrog(u0: np.ndarray, H: csr_matrix, nT: int, t_end: float, nSkip: int = 1, max_memory_gb: float = 1.0):
    nH, n_in, nFrames, dt, t, U = setup_time_evolution(u0, nT, t_end, nSkip, max_memory_gb)
    u_prev = U[0, 1:-1, 1:-1].ravel()
    u_curr = rk2_step(u_prev, H, dt)

    global_step = 1
    for k_frame in range(1, nFrames + 1):
        for _ in range(nSkip):
            u_next = leapfrog_step(u_curr, u_prev, H, dt, global_step)
            u_prev, u_curr = u_curr, u_next
            global_step += 1

        U_k = np.zeros((nH, nH), dtype=U.dtype)
        U_k[1:-1, 1:-1] = u_curr.reshape(n_in, n_in)
        U[k_frame] = U_k

    return t, U

def rk4(u0: np.ndarray, H: csr_matrix, nT: int, t_end: float, nSkip: int = 1, max_memory_gb: float = 1.0):
    nH, n_in, nFrames, dt, t, U = setup_time_evolution(u0, nT, t_end, nSkip, max_memory_gb)
    u = U[0, 1:-1, 1:-1].ravel()
    global_step = 1
    for k in range(1,nFrames+1):
        for _ in range(nSkip):
            u = rk4_step(u, H, dt,global_step)
            global_step += 1
        U_k = np.zeros((nH, nH), dtype=U.dtype)
        U_k[1:-1, 1:-1] = u.reshape(n_in, n_in)
        U[k] = U_k
    return t, U

def crank_nichelson(u0: np.ndarray, H: csr_matrix, nT: int, t_end: float, nSkip: int = 1, max_memory_gb: float = 1.0):
    nH, n_in, nFrames, dt, t, U = setup_time_evolution(u0, nT, t_end, nSkip, max_memory_gb)
    u = U[0, 1:-1, 1:-1].ravel()

    I = eye(n_in**2, format="csr")
    A = I + 0.5j * dt * H
    B = I - 0.5j * dt * H
    solveA = factorized(A)  # Finner LU faktorisering tidlig
    for k in range(1,nFrames+1):
        for i in range(nSkip):
            rhs = B @ u
            u = solveA(rhs)      # mye raskere enn spsolve(A, rhs) hver gang
        # Legger til randen
        U_k = np.zeros((nH, nH), dtype=U.dtype)
        U_k[1:-1, 1:-1] = u.reshape(n_in, n_in)
        U[k] = U_k
    return t, U



if __name__ == "__main__":
    v0 = 1e10
    nH = 100
    u0, X, Y = initialCondition(nH)
    V = v(X,Y,v0)
    H = build_H(nH,V)
    nT = 10_000_000
    nSkip = 10_000
    #t, U = finite_diff(u0,H,nT=nT,t_end=0.008,nSkip=nSkip)
    #t, U = leapfrog(u0,H,nT=nT,t_end=0.008,nSkip=nSkip)
    #t, U = rk4(u0,H,nT=nT,t_end=0.008,nSkip=nSkip)
    t, U = crank_nichelson(u0,H,nT=1000,t_end=0.008,nSkip=10)
    