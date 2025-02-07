import numpy as np
from scipy.optimize import minimize
from math import log as ln

height=5e-6
nu=0.5
baseline=0
angle=20
radius=2.5e-6
poc=0
ymod=1e3

def parabolic_fit(delta, E=ymod,  Zc=poc, dF=baseline, nu=nu, theta=angle):
    r"""Hertz model for a conical indenter

    """


    # root = Zc-delta
    # pos = root > Zc
    bb=np.zeros_like(delta)
    for i in range(len(delta)):
        if delta[i]<Zc:
            bb[i]=(1/np.sqrt(2))*((E*np.tan(theta))/(1-nu**2))*(delta[i]-Zc)**2
    return bb+dF

def cone(delta, E=ymod,  Zc=poc, Q=angle, dF=baseline, nu=nu):
    r"""Hertz model for a conical indenter"""
    Q=np.radians(Q)

    # root = Zc-delta
    # pos = root > Zc
    d=Zc-delta
    bb=np.zeros_like(delta)
    for i in range(len(delta)):
        if delta[i]<Zc:
            # bb[i]=(1/np.sqrt(2))*((E*np.tan(theta))/(1-nu**2))*(delta[i]-Zc)**2
            bb[i]=2/np.pi*E/(1-nu**2)*np.tan(Q)*(d[i])**2
    return bb+dF

def cone_bottom_herman(delta, E=ymod,  Zc=poc, h=height,Q=angle, dF=baseline, nu=nu):
    r"""Hertz model for a conical indenter with bottom effect correction"""
    Q=np.radians(Q)

    # root = Zc-delta
    # pos = root > Zc
    d=Zc-delta
    bb=np.zeros_like(delta)
    for i in range(len(delta)):
        if delta[i]<Zc:
            bb[i]=2/np.pi*E/(1-nu**2)*np.tan(Q)*(d[i])**2*(1+0.75*(np.tan(Q)*(d[i])/h)+0.609*(np.tan(Q)*(d[i])/h)**2+0.735*(np.tan(Q)*(d[i])/h)**3)
    return bb+dF

def hertz_r_free(delta, E,  Zc=poc, R=radius, dF=baseline, nu=nu):
    r"""Hertz model for a paraboloidal indenter"""
    #Q=np.radians(Q)
    d=Zc-delta
    # root = Zc-delta
    # pos = root > Zc
    bb=np.zeros_like(delta)
    for i in range(len(delta)):
        if delta[i]<Zc:
            bb[i]=(E/(1-nu**2))*(4/3)*np.sqrt(R)*((d[i])**(3/2))
            # print(rf'd: {d[i]}, zc={Zc}')
    return bb+dF

def parab_bott_herman(delta, E,  Zc=poc, h=height, R=radius, dF=baseline, nu=nu):
    r"""Hertz model for a paraboloidal indenter with bottom effect correction"""
    #Q=np.radians(Q)
    d=Zc-delta
    # root = Zc-delta
    # pos = root > Zc
    bb=np.zeros_like(delta)
    for i in range(len(delta)):
        if delta[i]<Zc:
            if d[i]-Zc<=0.4*h**2/R:
                bb[i]=(E/(1-nu**2))*(4/3)*np.sqrt(R)*((d[i])**(3/2))*(1+1.105*(R*(d[i])/h**2)**(1/2)+1.607*(R*(d[i])/h**2)+1.602*(R*(d[i])/h**2)**(3/2))
            else:
                bb[i]=(E/(1-nu**2))*(h**3/R)*(0.616-3.114*(R*(d[i])/h**2)**(1/2)+6.693*(R*(d[i])/h**2)-7.170*(R*(d[i])/h**2)**(3/2)+8.228*(R*(d[i])/h**2)**2+np.pi/2*(R*(d[i])/h**2)**3)
    return bb+dF



def FaSphereGO5good(delta, E, Zc, h, R, dF):
    r"""Approximate analytical formula of the contact mechanic model for an axisymmetric indenter for a thin film
    Gomila based on Dhaliwal and Rua's general solution """
    d=Zc-delta
    aa=get_a_G05good(delta, R, Zc, h)
    bb=np.zeros_like(delta)
    for i in range(len(delta)):
        if delta[i]<Zc:
            a=aa[i]
            bb[i]=  2*E*a/(1-0.5**2) * ((d[i]) * (1+1.12695*(a/h)+1.27003*(a/h)**2+0.61828*(a/h)**3-0.21941*(a/h)**4-0.48779*(a/h)**5) - 
                                        a*(2*R/a+(1-(R/a)**2)*ln((R+a)/(R-a))) * (1/4+0.281739*(a/h)+0.317507*(a/h)**2+0.256194*(a/h)**3+0.0596706*(a/h)**4-0.153756*(a/h)**5) + 
                                        a*(2*(R/(3*a)+(R/a)**3)+(1-(R/a)**4)*ln((R+a)/(R-a))) * (0.152434*(a/h)**3+0.171786*(a/h)**4+0.00797409*(a/h)**5) - 
                                        a*(2*(R/(5*a)+1/3*(R/a)**3+(R/a)**5)+(1-(R/a)**6)*ln((R+a)/(R-a))) * 0.0618736*(a/h)**5)
            # print(daSphereO4G(a,R,h),d[i]-Zc)
    return bb + dF
    


def daSphereO4G_G05good(a, R, h, Zc):
    if a<R:
        delta = (1/2 * a * ln((R + a) / (R - a)) +
                a * (2 * R / a + (1 - R**2 / a**2) * ln((R + a) / (R - a))) *
                (0.28173875 * (a / h) + 0.31750689 * (a / h)**2 + 0.0529485 * (a / h)**3 - 0.16937729 * (a / h)**4) +
                a * (2 * (R / (3 * a) + (R / a)**3) + (1 - (R / a)**4) * ln((R + a) / (R - a))) *
                (-0.152434 * (a / h)**3 - 0.171785 * (a / h)**4)) / \
                (1 + 1.126955 * (a / h) + 1.2700276 * (a / h)**2 - 0.19469572 * (a / h)**3 - 1.135605 * (a / h)**4)
    else:
        delta=np.nan
    return delta-Zc

def objective_G05good(a, delta, R, h, Zc):
    d=delta-Zc
    return abs((daSphereO4G_G05good(a, R, h, Zc) - d) * 100 / d)

def get_a_G05good(delta, R, Zc, h):
    """Compute the contact area radius (wrapper)"""
    d = Zc-delta
    aa = np.zeros_like(delta)
    bounds = [(1e-30, R*(1-1e-30))]
    for i in range(len(delta)):
        if delta[i] < Zc:
            a0 = [R / 2]
            result = minimize(objective_G05good, a0, args=(d[i], R, h, Zc), bounds=bounds, method='Nelder-Mead', options={'disp': False})            
            if result.success:
                aa[i] = result.x
            else:
                print(f"Optimization failed for index {i} with message: {result.message}")
    return aa

def FaSphereGomila(delta, E, Zc, h, R, dF, nu):
    r"""Approximate analytical formula of the contact mechanic model for an axisymmetric indenter for a thin film
    Gomila based on Dhaliwal and Rua's general solution """
    d=Zc-delta
    aa=get_a_gomila(delta, R, Zc, h)
    bb=np.zeros_like(delta)
    for i in range(len(delta)):
        if delta[i]<Zc:
            a=aa[i]
            bb[i]=  E/(1-nu**2)*((R**2+a**2)*np.arctanh(a/R)-a*R)*(1+0.8555*(a/h)**3)
    return bb + dF

def daSphereGomila(a, R, h, Zc):
    if a<R:
        delta = a*np.arctanh(a/R)+((R**2+a**2)*np.arctanh(a/R)-a*R)*(-0.5664*(1/h)-0.4846*a**3*(1/h)**4)+0.1069*((3*R**4+6*a**2*R**2+7*a**4)*np.arctanh(a/R)-a*R*(3*R**2+7*a**2))*(1/h)**3
    else:
        delta=np.nan
    return delta-Zc

def objective_gomila(a, delta, R, h, Zc):
    d=delta-Zc
    return abs((daSphereGomila(a, R, h, Zc) - d) * 100 / d)

def get_a_gomila(delta, R, Zc, h):
    """Compute the contact area radius (wrapper)"""
    d = Zc-delta
    aa = np.zeros_like(delta)
    bounds = [(1e-30, R*(1-1e-30))]
    for i in range(len(delta)):
        if delta[i] < Zc:
            a0 = [R / 2]
            result = minimize(objective_gomila, a0, args=(d[i], R, h, Zc), bounds=bounds, method='Nelder-Mead', options={'disp': False})            
            if result.success:
                aa[i] = result.x
            else:
                print(f"Optimization failed for index {i} with message: {result.message}")
    return aa

def FaSphereGomila_limit(delta, E, Zc, R, dF, nu):
    r"""Approximate analytical formula of the contact mechanic model for an axisymmetric indenter for a thin film
    Gomila based on Dhaliwal and Rua's general solution """
    d=Zc-delta
    aa=get_a_limit(delta, R, Zc)
    bb=np.zeros_like(delta)
    for i in range(len(delta)):
        if delta[i]<Zc:
            a=aa[i]
            # bb[i]=  E/(1-nu**2)*((R**2+a**2)*np.arctanh(a/R)-a*R)*(1+0.8555*(a/h)**3)
            bb[i]= E/(1-nu**2)*((1/2)*(a**2+R**2)*ln((R+a)/(R-a))-a*R)
    return bb + dF

def daSphereGomila_limit(a, R, Zc):
    if a<R:
        # delta = a*np.arctanh(a/R)+((R**2+a**2)*np.arctanh(a/R)-a*R)*(-0.5664*(1/h)-0.4846*a**3*(1/h)**4)+0.1069*((3*R**4+6*a**2*R**2+7*a**4)*np.arctanh(a/R)-a*R*(3*R**2+7*a**2))*(1/h)**3+Zc
        delta = (a/2)*ln((R+a)/(R-a))
    else:
        delta=np.nan
    return delta-Zc

def objective_limit(a, delta, R, Zc):
    d=delta-Zc
    return abs((daSphereGomila_limit(a, R, Zc) - d) * 100 / d)

def get_a_limit(delta, R, Zc):
    """Compute the contact area radius (wrapper)"""
    d = Zc-delta
    aa = np.zeros_like(delta)
    bounds = [(1e-30, R*(1-1e-30))]
    for i in range(len(delta)):
        if delta[i] < Zc:
            a0 = [R / 2]
            result = minimize(objective_limit, a0, args=(d[i], R, Zc), bounds=bounds, method='Nelder-Mead', options={'disp': False})            
            if result.success:
                aa[i] = result.x
            else:
                print(f"Optimization failed for index {i} with message: {result.message}")
    return aa