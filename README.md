# Procces_PIMDB_output
A python class dedicated for analysing LAMMPS output files.

To import the class:
from Read_Lammps_files import Analyze_LAMMPS

To use it:
MyAnalysis = Analyze_LAMMPS(bead#, url)

For example to calculate RDF:
MyAnalysis.RDF()

More examples in the "TestReport.ipynb" jupyter notebook
