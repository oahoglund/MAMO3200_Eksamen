"""
Hovedprogram som skal koble sammen alle de andre programmene.
Kaller på funksjoner fra simulering og visualisering.
Inneholder 'test' funksjoner for å manuelt sjekke om visse funksjoner funker.
Test funksjonene kan brukes for å teste programet ut, se hva som kan gjøres
"""
from simulering import *
from visualisering import *

def test_snap():
    v0 = 1e10
    nH = 100
    u0, X, Y = initialCondition(nH)
    V = v(X,Y,v0)
    p0 = convertToP(u0)
    V = v(X,Y,v0)
    #visualizeP_snapshot(p0,X,Y,V)
    visualizeP_snapshot(p0,X,Y,V, save=True, filename="init.png")

def test_v():
    v0 = 10
    nH = 100
    u0, X, Y = initialCondition(nH)
    V = v(X,Y,v0)
    print(V.shape)
    print(u0.shape)
    visualizeV(V,X,Y)

def test_anim():
    v0 = 1e6
    nH = 100
    u0, X, Y = initialCondition(nH)
    V = v(X,Y,v0)
    H = build_H(nH,V)
    #t, U = crank_nichelson(u0,H,nT=10000,t_end=0.008,nSkip=100)
    #t, U = finite_diff(u0,H,nT=10000,t_end=0.008,nSkip=100)
    t, U = rk4(u0,H,nT=10000,t_end=0.008,nSkip=100)
    P = convertToP(U)
    animateP(P,X,Y,t,V)

def test_anim_save():
    v0 = 1e4
    nH = 100
    u0, X, Y = initialCondition(nH)
    V = v(X,Y,v0)
    H = build_H(nH,V)
    t, U = crank_nichelson(u0,H,nT=10000,t_end=0.008,nSkip=100)
    P = convertToP(U)
    animateP_save(P,X,Y,t,V,gif_file="crank_nichelson_2.gif")

def test_totalP():
    v0 = 1e4
    nH = 100
    u0, X, Y = initialCondition(nH)
    V = v(X,Y,v0)
    H = build_H(nH,V)
    t, U = crank_nichelson(u0,H,nT=10,t_end=0.008,nSkip=1)
    p = totalPOvertime(U)
    visualizeTotalP(p,t,v0, "Crank Nichelson", "crank_nichelson_safe",save=True)

def test_totalPs():
    v0 = 1e4
    nH = 100
    u0, X, Y = initialCondition(nH)
    V = v(X,Y,v0)
    H = build_H(nH,V)
    nT = 10000
    nSkip=100
    t_end = 0.008
    t, U_fin = finite_diff(u0,H,nT=nT,t_end=t_end,nSkip=nSkip)
    _, U_leap = leapfrog(u0,H,nT=nT,t_end=t_end,nSkip=nSkip)
    _, U_rk4 = rk4(u0,H,nT=nT,t_end=t_end,nSkip=nSkip)
    p_fin = totalPOvertime(U_fin)
    p_leap = totalPOvertime(U_leap)
    p_rk4 = totalPOvertime(U_rk4)
    Ps  = {"Finite Difference" : p_fin, "Leapfrog": p_leap, "Runge Kutta 4": p_rk4}
    visualizeTotalPs(Ps,t)
    


if __name__ == "__main__":
    test_anim()
    #test_anim_save()
    #test_totalP()
    #test_totalPs()
    #test_v()
    #test_snap()