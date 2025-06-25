import numpy as np
from scipy.integrate import trapezoid, simpson, quad
from scipy.constants import c, G, pi
class Distances:
    #Classe per trovare le distanze cosmologiche data una funzione di Hubble
    
    #def __init__(self, hubble_function):
     #   self.hubble_function = hubble_function
    #class Distances:
    def __init__(self, H0, OmegaM, OmegaL, OmegaR=0):
        self.H0 = H0
        self.OmegaM = OmegaM
        self.OmegaL = OmegaL
        self.OmegaR = OmegaR

        # Definisci la funzione di Hubble standard come metodo della classe
        def hubble_function(z, w0=-1, wa=0):
            return self.H0 * np.sqrt(
                self.OmegaM * (1.0 + z)**3 +
                self.OmegaR * (1.0 + z)**4 +
                self.OmegaL
            )
        self.hubble_function = hubble_function
    
    def comoving_distance_trap(self, z):
        #calcola la distanza comoving in Mpc(float) dato un redshift
        xx = np.linspace(0, z, 1000000)
        fx= c/(1000*self.hubble_function(xx))
        Integral= trapezoid(fx, xx)
        return Integral
    
    def comoving_distance_simp(self, z):
        #calcola la distanza comoving in Mpc(float) dato un redshift
        xx = np.linspace(0, z, 1000000)
        fx= c/(1000*self.hubble_function(xx))
        Integral= simpson(fx, xx)
        return Integral
    
    def comoving_distance(self, z,w0=-1,wa=0):
        #calcola la distanza comoving in Mpc(float) dato un redshift
        def fx(x):
            return c/(1000*self.hubble_function(x,w0,wa))
        Integral=quad(fx, 0, z)[0]
        return Integral
    
    def angolar_distance(self, z):
        #calcola la distanza angolare in Mpc(float) dato un redshift
        return self.comoving_distance(z)/(1+z)
    
    def luminosity_distance(self, z):
        #calcola la distanza luminosa in Mpc(float) dato un redshift
        return self.comoving_distance(z)*(1+z)
    
    def v_distance(self, z, w0=-1,wa=0):
        #calcola la distanza luminosa in Mpc(float) dato un redshift
        return (z*self.comoving_distance(z,w0,wa)*self.comoving_distance(z,w0,wa)*c/(1000*self.hubble_function(z,w0,wa)))**(1/3)
    
    def m_distance(self, z, w0=-1,wa=0):
        #calcola la distanza luminosa in Mpc(float) dato un redshift
        return (self.comoving_distance(z,w0,wa))
        
    def h_distance(self, z, w0=-1,wa=0):
        #calcola la distanza luminosa in Mpc(float) dato un redshift
        return c/(1000*self.hubble_function(z,w0,wa))
    
    def distance_modulus(self, z):
        #calcola il modulo di distanza dato un redshift
        return 5*np.log10(self.luminosity_distance(z)*1e6/10)


