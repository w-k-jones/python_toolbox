import numpy as np

def latent_heat(T,phase=None):
    """
    Calcualtes the latent heat of condensation/deposition for liquid/ice water
    respectively using two empirical equations (Table 2.1. R. R. Rogers; M. K.
    Yau (1989). A Short Course in Cloud Physics (3rd ed.). Pergamon Press)
    Returns latent heat in J/g
    """
    scl_flag=False
    if np.isscalar(T):
        T = np.array([T])
        scl_flag = True
    # Check if in Kelvin and convert to Celsius
    T[T>100] -= 273.15


    # Initialise output array in shape of the temperature input
    out = np.full(T.shape, np.nan)

    # Find where the phase is liquid or T>0, and where the phase is ice or T<0
    wh_liq = np.logical_or(np.logical_and(phase=='liquid',T >= -25), T>0)
    wh_ice = np.logical_or(np.logical_or(np.logical_and(phase=='ice', T<=0), np.logical_and(phase!='liquid', T<=0)), T<-25)

    # Use empirical equations for liquid/ice latent heat of condensation/deposition
    out[wh_liq] = 2500.8-2.36*T[wh_liq]+0.0016*T[wh_liq]**2-0.00006*T[wh_liq]**3
    out[wh_ice] = 2834.1-0.29*T[wh_ice]-0.004*T[wh_ice]**2

    if scl_flag:
        out = out[0]

    return out
