#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:58:00 2023

@author: sagi

PLEASE MAKE SURE YOU HAVE INSTALLED: MDAnalysis & tqdm.
"""

import numpy as np
import pandas as pd
# import dask.array as da
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
import matplotlib.pyplot as plt
from tqdm import tqdm
# from tqdm.notebook import trange, tqdm
# from numba import njit
from functools import reduce
import glob
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Ignore the UserWarning
from copy import deepcopy


class Analyze_LAMMPS:
    
    def __init__(self, beads, filesPath):
        """
        

        Parameters
        ----------
        beads : INT
            NUMBER OF POLYMER BEADS.
        filesPath : STR
            THE FOLDER'S URL WHERE ALL THE OUTPUT FILES ARE.
            DON'T PLACE "/" AT THE END OF THE URL. 

        Returns
        -------
        None.

        """
        self.filesPath = filesPath
        self.beads = beads
        
        # for nice looking plots
        plt.rcParams['font.size'] = 15
        plt.rc('axes',linewidth=2,labelpad=10)
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rc('xtick.major',size=10, width=2)
        plt.rc('xtick.minor',size=7, width=2)
        plt.rc('ytick.major',size=10, width=2)
        plt.rc('ytick.minor',size=7, width=2)
        
    def find_logfile_read_position( self ):
        """
        FINDING THE ROWS AND FOOTERS TO SKIP IN "LOG.LAMMPS.*" FILES
        
        SETS self.skip_rows & self.skip_footer

        Returns
        -------
        None.

        """
        with open(self.filesPath+'/log.lammps.0','r') as file:
            fileList=file.readlines()
            for i in range(len(fileList)):
                if ('Per MPI rank memory allocation' in fileList[i]):
                    #print(i)
                    skip_rows=i+1
                if ('Loop time of' in fileList[i]):
                    #print(i)
                    skip_footer=len(fileList)-i
        self.skip_rows, self.skip_footer = skip_rows, skip_footer
    
    def avg_logfiles( self ):
        """
        READS ALL THE "LOG.LAMMPS.*" FILES. 
        
        SETS:
            "self.logFiles" AS A LIST OF DATA-FRAMES. FOR INSTANCE "df[5]" CORRESPONEDS TO "log.lammps.5"
            "self.AVG_logFiles" AS THE AVERAGE OF ALL THE LOG FILES.

        Returns
        -------
        None.

        """
        beads = self.beads
        self.find_logfile_read_position()
        sr, sf = self.skip_rows, self.skip_footer
        df = {}
        for i in tqdm(range(beads)):
            df[i]=pd.read_csv(self.filesPath+'/log.lammps.'+str(i),skiprows=sr,skipfooter=sf,engine='python',sep="\s+")
        df_total=reduce(lambda a, b: a.add(b, fill_value=0), [df[i] for i in range(beads)]) / beads
        
        self.logFiles, self.AVG_logFiles = df, df_total
        
    def position_tensor( self ):
        """
        READ FILES IN XYZ FORMAT

        Returns
        -------
        Position tensor: R[particle number, dimension, time step, bead number].

        """
        
        file_names = [i for i in glob.glob(self.filesPath+'/*.{}'.format('xyz'))]
        
        xyz_class = mda.coordinates.XYZ.XYZReader(file_names[0])
        steps = xyz_class.n_frames
        self.steps = steps
        particles = xyz_class.n_atoms
        beads = self.beads
        if ( beads!=len(file_names) ):
            print('error - number of xyz files does not equal to number of beads')
        
        R=np.zeros([particles,3,steps,beads])
        
        for ibead in tqdm(range(beads)):
            xyz_class = mda.coordinates.XYZ.XYZReader(file_names[ibead])
            for ts in xyz_class:
                R[:,:,ts.frame,ibead] = ts.positions
        return R
        
        
        
    def Radial_position_tensor( self ):
        """
        READ FILES IN XYZ FORMAT

        Returns
        -------
        Radial Position tensor: R[particle number, time step, bead number].

        """
        file_names = [i for i in glob.glob(self.filesPath+'/*.{}'.format('xyz'))]
        
        xyz_class = mda.coordinates.XYZ.XYZReader(file_names[0])
        steps = xyz_class.n_frames
        self.steps = steps
        particles = xyz_class.n_atoms
        beads = self.beads
        if ( beads!=len(file_names) ):
            print('error - number of xyz files does not equal to number of beads')
        
        R=np.zeros([particles,steps,beads])
        
        for ibead in tqdm(range(beads)):
            xyz_class = mda.coordinates.XYZ.XYZReader(file_names[ibead])
            for ts in xyz_class:
                R[:,ts.frame,ibead] = ( ( (ts.positions)**2 ).sum(1) )**0.5
        return R
    
    def density_Histogram( self, skip_frac=0.2, r_min=0, bins=100, output_name='density.csv', plot=False ):
        """
        GENERATES A CSV FILE WITH THE NUMBER OF COUNTS PER DISTANCE. ALSO RETURNS NORMELIZED DISTRIBUTION IN
        2D AND 3D SYSTEMS (CHOOSE THE RELEVANT FOR YOU). BTW, NORMELIZED MEANS THAT: \int \rho(r) d\omega dr = 1 .

        Parameters
        ----------
        skip_frac : fraction between 0 to 1, optional
            FRACTION OF DATA TO DROP AT THE BEGINING. The default is 0.2.
            
        r_min : non-negative float, optional
            THE MIN VALUE OF R. The default is 0.
            
        bins : positive integer, optional
            NUMBER OF BINS.
        
        output_name : string, optional
            THE NAME OF THE OUTPUT FILE
            
        plot : boolean, optional
            PLOTS A GRAPH OF THE DENSITY. The default is not to plot.
        
        Returns
        -------
        None.

        """
        R = self.Radial_position_tensor()
        drop = int(skip_frac*self.steps)
        R = R.copy()[:,drop:,:]
        counts,R_range=np.histogram(R[R>r_min],bins)
        R_range=(R_range[1:]+R_range[:-1])*0.5
        
        counts_Area = np.trapz(counts, R_range)
        norm_2D_counts = counts/R_range/counts_Area/(2*np.pi)
        norm_3D_counts = counts/R_range/R_range/counts_Area/(4*np.pi)
        
        density_dict = {'R' : R_range, 'counts' : counts, '2D_counts': norm_2D_counts, '3D_counts': norm_3D_counts}
        df = pd.DataFrame(density_dict)
        df.to_csv(output_name,index=False)
        
        if plot:
            fig, ax = plt.subplots(figsize=(5,3))
            ax.plot(R_range,counts)
            ax.set_xlabel(r'r', labelpad=10)
            ax.set_ylabel(r'$\rho$(r)', labelpad=10)
        
    
    def RDF( self, atom_A='all', atom_B='all',skip_frac=0.2, r_min=0, r_max=400, bins=100, output_name='rdf.csv', plot=False ):
        """
        GENERATES A CSV FILE OF THE PAIR CORRELATION. ALSO RETURNS NORMELIZED DISTRIBUTION IN
        2D AND 3D SYSTEMS (CHOOSE THE RELEVANT FOR YOU). BTW, NORMELIZED MEANS THAT: \int \g(r) d\omega dr = 1 .

        Parameters
        ----------
        atom_A/B : string, optional
            IN CASE THAT THERE ARE DIFFERENT KINDS OF ATOMS.
            FOR EXAMPLE: RDF OF WATER BETWEEN ATOM O AND H: atom_A = 'type O', atom_B = 'type H'

        skip_frac : (currently not implemented) fraction between 0 to 1, optional
            FRACTION OF DATA TO DROP AT THE BEGINING. The default is 0.2.
            
        r_min : non-negative float, optional
            THE MIN VALUE OF R. The default is 0.
            
        bins : positive integer, optional
            NUMBER OF BINS.
        
        output_name : string, optional
            THE NAME OF THE OUTPUT FILE
            
        plot : boolean, optional
            PLOTS A GRAPH OF THE DENSITY. The default is not to plot.
        
        Returns
        -------
        None.

        """
        
        file_names = [i for i in glob.glob(self.filesPath+'/*.{}'.format('xyz'))]
        
        xyz_class = mda.coordinates.XYZ.XYZReader(file_names[0])
        steps = xyz_class.n_frames
        self.steps = steps
        drop=int(skip_frac*self.steps)
        particles = xyz_class.n_atoms
        beads = self.beads
        if ( beads!=len(file_names) ):
            print('error - number of xyz files does not equal to number of beads')
        
        counts,Rij_range = 0,0
        for ibead in tqdm(range(beads)):
            try:
                u= mda.Universe(file_names[ibead])
                # atoms = u.select_atoms('all')
                A = u.select_atoms(atom_A)
                B = u.select_atoms(atom_B)
                irdf = rdf.InterRDF(A, B,exclusion_block=(1, 1),nbins=bins,range=(r_min,r_max),norm=None)
                irdf.run()
            except UserWarning:
                pass
            counts+=irdf.results.rdf
            Rij_range+=irdf.results.bins/beads
            
        
        counts_Area = np.trapz(counts, Rij_range)
        norm_2D_counts = counts/Rij_range/counts_Area/(2*np.pi)
        norm_3D_counts = counts/Rij_range/Rij_range/counts_Area/(4*np.pi)
        
        density_dict = {'Rij' : Rij_range, 'counts' : counts, '2D_counts': norm_2D_counts, '3D_counts': norm_3D_counts}
        df = pd.DataFrame(density_dict)
        df.to_csv(output_name,index=False)
        
        if plot:
            fig, ax = plt.subplots(figsize=(5,3))
            ax.plot(Rij_range,counts)
            ax.set_xlabel(r'r', labelpad=10)
            ax.set_ylabel(r'g(r)', labelpad=10)
        
        
    def ImCorrelation(self, skip_frac=0.2, beta=1, hbar=1, mass=1, dim=3, output_name='G.csv',pbc=None):
        '''
        GENERATES A CSV FILE OF IMAGINARY TIME CORRELATION FUNCTION
        '''
        
        
        file_names = [i for i in glob.glob(self.filesPath+'/*.{}'.format('xyz'))]
        # to rearrange files according to bead number
        file_names = sorted(file_names, key=lambda x: int(x.split('_')[1].split('.')[0]))
        xyz_class = mda.coordinates.XYZ.XYZReader(file_names[0])
        steps = xyz_class.n_frames
        self.steps = steps
        particles = xyz_class.n_atoms
        beads = self.beads
        drop=int(skip_frac*self.steps)
        if ( beads!=len(file_names) ):
            print('error - number of xyz files does not equal to number of beads')

        tau_k = np.linspace(0,beta,beads+1)
        G = np.zeros([beads+1]) #G = da.zeros([beads+1])
        R_0 = np.zeros([particles,3,steps]) #R_0 = da.zeros([particles,3,steps])
        R_1 = np.zeros([particles,3,steps]) #R_1 = da.zeros([particles,3,steps])
        
        xyz_class2 = mda.coordinates.XYZ.XYZReader(file_names[1])
        for ts,ts2 in zip(xyz_class,xyz_class2):
            #arr1 = da.from_array(ts.positions)
            #arr2 = da.from_array(ts2.positions)
            R_0[:,:,ts.frame] = ts.positions
            R_1[:,:,ts.frame] = ts2.positions
        R_0_1 = deepcopy(R_1)[:,:,drop:] - deepcopy(R_0)[:,:,drop:]
        if pbc is not None:
            R_0_1[R_0_1>=pbc/2] -= pbc
            R_0_1[R_0_1<-pbc/2] += pbc

        G[0] = beads*dim/(mass*beta) - (beads*beads/(hbar*beta)**2) * ( (R_0_1*R_0_1).sum() ) /(steps-drop)/particles
        # G[0] =  - (beads*beads/(hbar*beta)**2) * ( (R_0_1*R_0_1).sum() )/(steps-drop)/particles
        
        R_k = deepcopy(R_1)
        for ibead in tqdm(range(2,beads)):
            R_k1 = np.zeros([particles,3,steps])
            xyz_class = mda.coordinates.XYZ.XYZReader(file_names[ibead])
            for ts in xyz_class:
                #arr=da.from_array(ts.positions)
                R_k1[:,:,ts.frame] = ts.positions
            R_k_k1 = deepcopy(R_k1)[:,:,drop:] - deepcopy(R_k)[:,:,drop:]
            if pbc is not None:
                R_k_k1[R_k_k1>=pbc/2] -= pbc
                R_k_k1[R_k_k1<-pbc/2] += pbc
            G[ibead-1] = - (beads*beads/(hbar*beta)**2) * ( (R_k_k1*R_0_1).sum() ) /(steps-drop)/particles
            R_k = deepcopy(R_k1)
        R_k_k1 = deepcopy(R_0)[:,:,drop:] - deepcopy(R_k)[:,:,drop:]
        if pbc is not None:
            R_k_k1[R_k_k1>=pbc/2] -= pbc
            R_k_k1[R_k_k1<-pbc/2] += pbc

        G[ibead] = - (beads*beads/(hbar*beta)**2) * ( (R_k_k1*R_0_1).sum() ) /(steps-drop)/particles
        G[ibead+1] = G[0]

        #G = G.compute()

        dict = {'tau' : tau_k, 'G' : G}
        df = pd.DataFrame(dict)
        df.to_csv(output_name,index=False)

    def ImCorrelation_molecule(self, skip_frac=0.2, beta=1, hbar=1, use_kg_mass_units=True, dim=3, output_name='G.csv',pbc=None):
        '''
        GENERATES A CSV FILE OF IMAGINARY TIME CORRELATION FUNCTION FOR MOLECULES
        '''
        
        
        file_names = [i for i in glob.glob(self.filesPath+'/*.{}'.format('xyz'))]
        # to rearrange files according to bead number
        file_names = sorted(file_names, key=lambda x: int(x.split('_')[1].split('.')[0]))
        xyz_class = mda.coordinates.XYZ.XYZReader(file_names[0])
        steps = xyz_class.n_frames
        self.steps = steps
        particles = xyz_class.n_atoms
        beads = self.beads
        drop=int(skip_frac*self.steps)
        if ( beads!=len(file_names) ):
            print('error - number of xyz files does not equal to number of beads')
        
        # The mass vector of the atoms
        u = mda.Universe(file_names[0])
        masses = u.atoms.masses # the units are gram/mol
        if use_kg_mass_units:
            masses /= (1000 * 6.02214076 * 1E+23) # now in kg

        tau_k = np.linspace(0,beta,beads+1)
        G = np.zeros([beads+1])
        R_0 = np.zeros([particles,3,steps]) 
        R_1 = np.zeros([particles,3,steps]) 
        
        xyz_class2 = mda.coordinates.XYZ.XYZReader(file_names[1])
        for ts,ts2 in zip(xyz_class,xyz_class2):
            R_0[:,:,ts.frame] = ts.positions
            R_1[:,:,ts.frame] = ts2.positions
        R_0_1 = deepcopy(R_1)[:,:,drop:] - deepcopy(R_0)[:,:,drop:]
        if pbc is not None:
            R_0_1[R_0_1>=pbc/2] -= pbc
            R_0_1[R_0_1<-pbc/2] += pbc

        G[0] =( beads*dim/(masses*beta) ).mean() - (beads*beads/(hbar*beta)**2) * ( (R_0_1*R_0_1).sum() ) /(steps-drop)/particles
        
        R_k = deepcopy(R_1)
        for ibead in tqdm(range(2,beads)):
            R_k1 = np.zeros([particles,3,steps])
            xyz_class = mda.coordinates.XYZ.XYZReader(file_names[ibead])
            for ts in xyz_class:
                R_k1[:,:,ts.frame] = ts.positions
            R_k_k1 = deepcopy(R_k1)[:,:,drop:] - deepcopy(R_k)[:,:,drop:]
            if pbc is not None:
                R_k_k1[R_k_k1>=pbc/2] -= pbc
                R_k_k1[R_k_k1<-pbc/2] += pbc
            G[ibead-1] = - (beads*beads/(hbar*beta)**2) * ( (R_k_k1*R_0_1).sum() ) /(steps-drop)/particles
            R_k = deepcopy(R_k1)
        R_k_k1 = deepcopy(R_0)[:,:,drop:] - deepcopy(R_k)[:,:,drop:]
        if pbc is not None:
            R_k_k1[R_k_k1>=pbc/2] -= pbc
            R_k_k1[R_k_k1<-pbc/2] += pbc

        G[ibead] = - (beads*beads/(hbar*beta)**2) * ( (R_k_k1*R_0_1).sum() ) /(steps-drop)/particles
        G[ibead+1] = G[0]

        dict = {'tau' : tau_k, 'G' : G}
        df = pd.DataFrame(dict)
        df.to_csv(output_name,index=False)
        
