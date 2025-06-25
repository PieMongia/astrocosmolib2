import numpy as np
import Corrfunc

class TwoPointIsotropic:

    """
    Class to compute the isotropic two-point correlation function
    """

    def __init__(self):
        """
        Initialize the TwoPointIsotropic class
        """
        pass

    def set_separation(self, r_min, r_max, delta_r):
        """
        Set the separation bins for the two-point correlation function

        Parameters
        ----------
        r_min : float
            Minimum separation
        r_max : float
            Maximum separation
        delta_r : float
            Bin width
        """
        self.r_min = r_min
        self.r_max = r_max
        self.delta_r = delta_r

        # Create the bin edges and centers
        self.r_edges = np.arange(r_min, r_max + delta_r, delta_r)
        self.r_centers = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])
        self.n_bins = len(self.r_centers)
    
    def compute_auto_pairs(self, xcoords, ycoords, zcoords, weights=None, periodic=True, boxsize=None, nthreads=1):
        """
        Compute the two-point correlation function for a single set of points

        Parameters
        ----------
        xcoords : array_like
            x-coordinates of the points
        ycoords : array_like
            y-coordinates of the points
        zcoords : array_like
            z-coordinates of the points
        weights : array_like, optional
            Weights for each point (default is None)
        periodic : bool, optional
            Whether to use periodic boundary conditions (default is True)
        boxsize : float, optional
            Size of the box (default is None)
        nthreads : int, optional
            Number of threads to use (default is 1)
        Returns
        -------
        r : array_like
            Separation bins
        xi : array_like
            Two-point correlation function values
        """
        if periodic and boxsize is None:
            raise ValueError("Box size must be provided for periodic boundary conditions")
        
        weight_type = None
        if weights is not None:
            weight_type = "pair_product"

        # Compute the two-point correlation function using Corrfunc
        result = Corrfunc.theory.DD(True,
            nthreads=1,
            binfile=self.r_edges,
            X1=xcoords,
            Y1=ycoords,
            Z1=zcoords,
            weights1=weights,
            periodic=periodic,
            boxsize=boxsize,
            weight_type=weight_type
        )

        return result

    def compute_cross_pairs(self, x1coords, y1coords, z1coords, x2coords, y2coords, z2coords, weights1=None, weights2=None, periodic=True, boxsize=None, nthreads=1):
        """
        Compute the two-point correlation function for a single set of points

        Parameters
        ----------
        x1coords : array_like
            x-coordinates of the points in the first set
        y1coords : array_like
            y-coordinates of the points in the first set
        z1coords : array_like
            z-coordinates of the points in the first set
        x2coords : array_like
            x-coordinates of the points in the second set
        y2coords : array_like
            y-coordinates of the points in the second set
        z2coords : array_like
            z-coordinates of the points in the second set
        weights1 : array_like, optional
            Weights for each point in the first set (default is None)
        weights1 : array_like, optional
            Weights for each point in the second set (default is None)
        periodic : bool, optional
            Whether to use periodic boundary conditions (default is True)
        boxsize : float, optional
            Size of the box (default is None)
        nthreads : int, optional
            Number of threads to use (default is 1)
        Returns
        -------
        r : array_like
            Separation bins
        xi : array_like
            Two-point correlation function values
        """
        if periodic and boxsize is None:
            raise ValueError("Box size must be provided for periodic boundary conditions")
        
        weight_type = None
        if (weights1 is not None) or (weights2 is not None):
            weight_type = "pair_product"

        # Compute the two-point correlation function using Corrfunc
        result = Corrfunc.theory.DD(autocorr=False,
            nthreads=1,
            binfile=self.r_edges,
            X1=x1coords,
            Y1=y1coords,
            Z1=z1coords,
            weights1=weights1,
            X2=x2coords,
            Y2=y2coords,
            Z2=z2coords,
            weights2=weights2,
            periodic=periodic,
            boxsize=boxsize,
            weight_type=weight_type
        )

        return result

    def compute(self, 
                r_min,
                r_max,
                delta_r,
                x1coords,
                y1coords,
                z1coords,
                x2coords,
                y2coords,
                z2coords,
                weights1=None,
                weights2=None,
                periodic=True,
                boxsize=None,
                nthreads=1):
        """
        Compute the two-point correlation function
        """

        # Set the separation bins
        self.set_separation(r_min, r_max, delta_r)

        # Compute DD
        self.DD = self.compute_auto_pairs(x1coords,
                                          y1coords,
                                          z1coords,
                                          weights=weights1,
                                          periodic=periodic,
                                          boxsize=boxsize,
                                          nthreads=nthreads)    

        # Compute RR
        self.RR = self.compute_auto_pairs(x2coords,
                                          y2coords,
                                          z2coords,
                                          weights=weights2,
                                          periodic=periodic,
                                          boxsize=boxsize,
                                          nthreads=nthreads)
        
        # Compute DR
        self.DR = self.compute_cross_pairs(x1coords,
                                           y1coords,
                                           z1coords,
                                           x2coords,
                                           y2coords,
                                           z2coords,
                                           weights1=weights1,
                                           weights2=weights2,
                                           periodic=periodic,
                                           boxsize=boxsize,
                                           nthreads=nthreads)

        # Compute the two-point correlation function

        # Get pairs
        DD = self.DD['npairs'] 
        RR = self.RR['npairs']
        DR = self.DR['npairs']

        Nd = len(x1coords)**2
        Nr = len(x2coords)**2
        Ndr = len(x1coords) * len(x2coords)

        if (weights1 is not None): 
            DD *= self.DD['weightavg']
            Nd = np.sum(weights1)**2
        
        if (weights2 is not None):
            RR *= self.RR['weightavg']
            Nr = np.sum(weights1)**2

        if (weights1 is not None) and (weights2 is not None):
            DR *= self.DR['weightavg']
            Ndr = np.sum(weights1) * np.sum(weights2)

        # Compute the two-point correlation function
        self.xi = (DD/Nd + RR/Nr - 2*DR/Ndr) / (RR/Nr)