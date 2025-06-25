from astropy.io import fits
import numpy as np

from ..helpers.logger import Logger

class FitsManager:
    """
    A class to manage fits files, using astropy.io.fits module
    """

    #define the costructor
    def __init__(self, input_file):
        """
        Constructor of the class
        :param input_file: the path of the fits file to be managed
        """
        self.input_file = input_file
        self.hdulist = fits.open(input_file)

        #create the logger
        self.logger = Logger("FitsManager")
        self.logger("Fits file opened successfully ")

    def get_header(self, hdu_index):
        """
        Get the header of a given HDU
        :param hdu: the index of the HDU
        :return: the header of the HDU
        """
        if hdu_index < 0 or hdu_index > len(self.hdulist):
            self.logger.error("Invalid HDU index",ValueError)
            raise ValueError("Invalid HDU index")
        
        return self.hdulist[hdu_index].header
    
    def get_hdu_count(self, hdu_index):        
        """
        Get the number of HDUs in the fits file
        :return: the number of HDUs
        """
        if hdu_index < 0 or hdu_index > len(self.hdulist):
            self.logger.error("Invalid HDU index",ValueError)
            raise ValueError("Invalid HDU index")
        
        return len(self.hdulist)
    
    def get_data(self, hdu_index):
        """
        Get the data of a given HDU
        :param hdu: the index of the HDU
        :return: the data of the HDU
        """
        if hdu_index < 0 or hdu_index > len(self.hdulist):
            self.logger.error("Invalid HDU index",ValueError)
            raise ValueError("Invalid HDU index")
        
        return self.hdulist[hdu_index].data
    
    # python -m astrocosmolib.pyACC.fits_wrapper.fits_manager