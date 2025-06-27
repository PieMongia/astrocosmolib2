import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfftn, fftfreq, rfftfreq
import Corrfunc
import camb 
from mcfit import P2xi

def calculate_power_spectrum_fft(positions, box_size, n_cells=64):
    """Calcola P(k) dal campo di densità usando la FFT."""
    print("... calcolo P(k) con FFT...")

    #Calcolo la densità media
    n_points = len(positions)
    density = n_points / box_size**3
    
    #Creazione del campo di densità usando histogramdd che divide il box in celle e conta il numero di punti
    grid, edges = np.histogramdd(positions, bins=(n_cells, n_cells, n_cells), range=((0, box_size), (0, box_size), (0, box_size)))
    cell_mean_density = n_points / n_cells**3
    density_field = (grid - cell_mean_density) / cell_mean_density
    
    #Faccio la FFT del campo di densità e calcolo il Power Spectrum
    delta_k = rfftn(density_field) / n_cells**3
    pk_grid = np.abs(delta_k)**2 * (box_size**3)
    
    #Calcolo i vettori d'onda k
    kF = 2 * np.pi / box_size
    kx = fftfreq(n_cells) * n_cells * kF
    ky = fftfreq(n_cells) * n_cells * kF
    kz = rfftfreq(n_cells) * n_cells * kF
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    KK = np.sqrt(KX**2 + KY**2 + KZ**2)
    
    #Faccio il binning del Power Spectrum
    k_edges = np.arange(kF, np.pi * n_cells / box_size, kF)
    k_bins = 0.5 * (k_edges[1:] + k_edges[:-1])
    pk_average, _ = np.histogram(KK.flatten(), bins=k_edges, weights=pk_grid.flatten())
    counts, _ = np.histogram(KK.flatten(), bins=k_edges)
    
    #Normalizzo il Power Spectrum e correggo per il rumore
    pk_average = np.divide(pk_average, counts, out=np.full_like(pk_average, np.nan), where=counts!=0)
    pk_corrected = pk_average - (1.0 / density)
    
    return k_bins, pk_corrected

#FUNZIONE DI CALCOLO DELLA FUNZIONE DI CORRELAZIONE XI(R) USAndo CORRFUNC E IL METODO LANDY-SZALAY

def calculate_correlation_function(positions, box_size, n_random_factor=5, n_threads=4):
    n_data = len(positions)
    
    print(f"... calcolo xi(r) per {n_data} punti...")
    positions = np.asarray(positions, dtype=np.float64)

    # Crea catalogo casuale
    n_random = n_data * n_random_factor
    random_positions = np.random.uniform(0, box_size, size=(n_random, 3))

    # Definisci i bin e calcola i loro centri
    r_bins = np.logspace(0, 2.3, 50)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:]) # Centro geometrico dei bin

    # Calcola DD (Data-Data pairs)
    dd_res = Corrfunc.theory.DD(autocorr=True, nthreads=n_threads, binfile=r_bins,
                               X1=positions[:, 0], Y1=positions[:, 1], Z1=positions[:, 2],
                               boxsize=box_size, periodic=True)

    # Calcola DR (Data-Random pairs)
    dr_res = Corrfunc.theory.DD(autocorr=False, nthreads=n_threads, binfile=r_bins,
                               X1=positions[:, 0], Y1=positions[:, 1], Z1=positions[:, 2],
                               X2=random_positions[:, 0], Y2=random_positions[:, 1], Z2=random_positions[:, 2],
                               boxsize=box_size, periodic=True)

    # Calcola RR (Random-Random pairs)
    rr_res = Corrfunc.theory.DD(autocorr=True, nthreads=n_threads, binfile=r_bins,
                               X1=random_positions[:, 0], Y1=random_positions[:, 1], Z1=random_positions[:, 2],
                               boxsize=box_size, periodic=True)

    # Estrai conteggi
    DD = dd_res['npairs']
    DR = dr_res['npairs']
    RR = rr_res['npairs']

    # Fattori di normalizzazione
    norm_dd = n_data * n_data
    norm_dr = n_data * n_random
    norm_rr = n_random * n_random

    # Formula di Landy-Szalay
    with np.errstate(invalid='ignore', divide='ignore'):
        xi = ( (DD / norm_dd) - 2 * (DR / norm_dr) + (RR / norm_rr) ) / (RR / norm_rr)

    xi[np.isnan(xi)] = 0.0 
    return r_centers, xi

def get_theoretical_pk_camb(cosmo_params, redshift):

    print(f"... calcolo P(k) teorico con CAMB per z={redshift}...")
    
    # Lista dei redshift da calcolare: sempre z=0 e il redshift target (se diverso)
    target_redshift = float(redshift)
    redshifts_to_calc = [0.0]
    if not np.isclose(target_redshift, 0.0):
        redshifts_to_calc.append(target_redshift)
    
    camb_params = camb.CAMBparams()

    #Setto la cosmologia in CAMB
    camb_params.set_cosmology(
        H0=cosmo_params['H0'], 
        ombh2=cosmo_params['ombh2'], 
        omch2=cosmo_params['omch2'],
        mnu=cosmo_params['mnu'],
        num_massive_neutrinos=cosmo_params['num_massive_neutrinos']
    )
    camb_params.InitPower.set_params(ns=cosmo_params['ns'])
    # Imposto CAMB per calcolare P(k) a tutti i redshift nella lista
    camb_params.set_matter_power(redshifts=sorted(list(set(redshifts_to_calc))), kmax=10.0)    
    camb_results = camb.get_results(camb_params)
    sigma8_z0_camb = camb_results.get_sigma8_0()
    renorm_factor = (cosmo_params['sigma8_target'] / sigma8_z0_camb)**2  
    # Prendi lo spettro di potenza per il nostro redshift di interesse
    k_theory, z_array, pk_theory_raw = camb_results.get_matter_power_spectrum(minkh=1e-4, maxkh=10.0, npoints=200)
    
    # Seleziona il P(k) giusto (quello al nostro redshift target)
    pk_for_target_z = camb_results.get_matter_power_interpolator().P(z_array, k_theory)[-1]
    
    pk_theory_renorm = pk_for_target_z * renorm_factor
    
    return k_theory, pk_theory_renorm

#FUNZIONE PER CALCOLARE xi(r) TEORICO DA P(k) 

#Utilizza la libreria 'mcfit' per la trasformata di Fourier
def get_theoretical_xi_from_pk(k, pk, bias):
    if P2xi is None: return None, None
    print("... calcolo xi(r) teorica da P(k)...")

    # Applica il bias al P(k) della materia
    pk_biased = pk * (bias**2)
    
    # Esegui la trasformata di Fourier per ottenere xi(r)
    r, xi = P2xi(k)(pk_biased)
    return r, xi