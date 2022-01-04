#!/usr/bin/env python3

'''
MDAnalysis - protein RMSF calculation

'''

import numpy as np
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF, DCD
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import RMSF


def calc_protalign(aa):
    protein = aa.select_atoms('protein')
    select = 'protein and name CA'

    # superimpose the frames based on selected region of protein,
    # which will be used to calculate rmsf
    prealigner = align.AlignTraj(aa, aa,
                                 select=select,
                                 in_memory=True).run()

    ref_coordinates = aa.trajectory.timeseries(asel=protein).mean(axis=1)

    ref = mda.Merge(protein).load_new(ref_coordinates[:, None, :], order="afc")

    aligner = align.AlignTraj(aa, ref,
                              select=select,
                              in_memory=True).run()

    return aa, protein


def calc_rmsf(aa):
    # pass through align
    aa, protein = calc_protalign(aa)

    protein_sel = 'name CA'

    # get rmsf per residue of selected protein region
    # based on the average fluctuations
    protein_region = protein.select_atoms(protein_sel)

    rmsfer = RMSF(protein_region).run()
    traj_rmsf = np.array(rmsfer.rmsf)

    # return protein_region, rmsfer
    return protein_region, traj_rmsf


def main():
    # load test trajectory
    aa = mda.Universe(PSF, DCD)

    # calculate RMSF
    # protein_region, rmsfer = calc_rmsf(aa)
    protein_region, traj_rmsf = calc_rmsf(aa)

    # plot
    # plt.plot(protein_region.resnums, rmsfer.rmsf)
    plt.plot(protein_region.resnums, traj_rmsf,
             color='darkgreen',
             alpha=0.8)

    plt.xlabel('residue number')
    plt.ylabel('RMSF ($\AA$)')
    plt.savefig('rmsf_plot.png')
    plt.show()


if __name__ == '__main__':
    main()
