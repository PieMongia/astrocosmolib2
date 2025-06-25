import numpy as np

class LambdaCDM:
    """
    LambdaCDM cosmology model.
    This class implements the LambdaCDM cosmology model, which includes
    matter, radiation, and dark energy components.
    """

    def __init__(self, H0, Omega_m, Omega_Lambda, Omega_radiation=0.0):
        """
        Initialize the LambdaCDM cosmology model.
        Parameters
        ----------
        H0 : float
            Hubble constant at z=0 in km/s/Mpc
        Omega_m : float
            Matter density parameter at z=0
        Omega_Lambda : float
            Dark energy density parameter at z=0
        Omega_radiation : float, optional
            Radiation density parameter at z=0 (default is 0.0)
        """


        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_Lambda = Omega_Lambda
        self.Omega_radiation = Omega_radiation
        self.Omega_k = 1.0 - (self.Omega_m + self.Omega_Lambda + self.Omega_radiation)
    
    def E(self, z):
        """
        Dimensionless Hubble parameter at redshift z.
        Parameters
        ----------
        z : float
            Redshift
        Returns
        -------
        float
            Dimensionless Hubble parameter
        """

        return np.sqrt(
            self.Omega_m * (1 + z)**3 +
            self.Omega_radiation * (1 + z)**4 +
            self.Omega_k * (1 + z)**2 +
            self.Omega_Lambda
        )
    
    def H(self, z):
        """
        Hubble parameter at redshift z.
        Parameters
        ----------
        z : float
            Redshift
        Returns
        -------
        float
            Hubble parameter in km/s/Mpc
        """

        return self.H0 * self.E(z)

class FlatLambdaCDM:
    """
    FlatLambdaCDM cosmology model.
    This class implements the FlatLambdaCDM cosmology model, which includes
    matter, radiation, and dark energy components.
    """

    def __init__(self, H0, Omega_m, Omega_radiation=0.0):
        """
        Initialize the LambdaCDM cosmology model.
        Parameters
        ----------
        H0 : float
            Hubble constant at z=0 in km/s/Mpc
        Omega_m : float
            Matter density parameter at z=0
        Omega_radiation : float, optional
            Radiation density parameter at z=0 (default is 0.0)
        """


        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_radiation = Omega_radiation
        self.Omega_k = 0
        self.Omega_Lambda = 1 - self.Omega_m - self.Omega_radiation
    
    def E(self, z):
        """
        Dimensionless Hubble parameter at redshift z.
        Parameters
        ----------
        z : float
            Redshift
        Returns
        -------
        float
            Dimensionless Hubble parameter
        """

        return np.sqrt(
            self.Omega_m * (1 + z)**3 +
            self.Omega_radiation * (1 + z)**4 +
            self.Omega_Lambda
        )
    
    def H(self, z):
        """
        Hubble parameter at redshift z.
        Parameters
        ----------
        z : float
            Redshift
        Returns
        -------
        float
            Hubble parameter in km/s/Mpc
        """

        return self.H0 * self.E(z)

class w0waFlatCDM:
    """
    w0waCDM cosmology model.
    This class implements the w0waCDM cosmology model, which includes
    matter, radiation, and dark energy components with a time-varying
    equation of state.
    """

    def __init__(self, H0, Omega_m, Omega_radiation=0.0, w0=-1, wa=0):
        """
        Initialize the w0waFlatCDM cosmology model.
        Parameters
        ----------
        H0 : float
            Hubble constant at z=0 in km/s/Mpc
        Omega_m : float
            Matter density parameter at z=0
        Omega_radiation : float, optional
            Radiation density parameter at z=0 (default is 0.0)
        w0 : float, optional
            Equation of state parameter at z=0 (default is -1)
        wa : float, optional
            Time variation of the equation of state parameter (default is 0)
        """


        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_DarkEnergy = 1 - Omega_m - Omega_radiation
        self.Omega_radiation = Omega_radiation
        self.Omega_k = 0
        self.w0 = w0
        self.wa = wa
        self.Omega_k = 0
    
    def E(self, z):
        """
        Dimensionless Hubble parameter at redshift z.
        Parameters
        ----------
        z : float
            Redshift
        Returns
        -------
        float
            Dimensionless Hubble parameter
        """

        return np.sqrt(
            self.Omega_m * (1 + z)**3 +
            self.Omega_radiation * (1 + z)**4 +
            self.Omega_DarkEnergy * (1 + z)**(3 * (1 + self.w0 + self.wa)) * np.exp(-3 * self.wa * z/(1 + z))
        )
    
    def H(self, z):
        """
        Hubble parameter at redshift z.
        Parameters
        ----------
        z : float
            Redshift
        Returns
        -------
        float
            Hubble parameter in km/s/Mpc
        """

        return self.H0 * self.E(z)