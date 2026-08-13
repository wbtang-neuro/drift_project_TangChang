+-------------------------------------------------------------------------+
|        Circuit mechanisms for disentangled representations and          | 
|                     generalization in the hippocampus                   |
+-------------------------------------------------------------------------+
README.txt
Copyright (C) 2026, Wenbo Tang, version 1.0
All rights reserved.

DESCRIPTION OF THE CODE CONTAINED IN THE ARCHIVE: drift_project.tgz


BRIEF
=====

Code accompanying the paper: Tang, W.*, Chang, H.*, et al. Nature Neuroscience (2026). Circuit mechanisms for disentangled representations and generalization in the hippocampus.


GETTING STARTED
===============

Launch MATLAB and cd into the directory containing the code (e.g., '/drift_project/').

Other files in the directory (with all sub-folders) needed in the path:  
https://github.com/ayalab1/neurocode

Toolboxes required:
- For UMAP: Uniform Manifold Approximation and Projection (UMAP) (version 4.1; https://www.mathworks.com/matlabcentral/fileexchange/71902-uniform-manifold-approximation-and-projection-umap)

- for homology: PersistentHomology (https://github.com/codetyt/PersistentHomologyOnMATLAB)

- for Reduced Rank Regression (RRR): communication-subspace-master  (https://github.com/joao-semedo/communication-subspace)

- for Canonical Correlation Analysis (CCA): canonical-correlation-maps-main  (https://github.com/joao-semedo/canonical-correlation-maps)

- for hippocampus network models: representation-drift-main  (https://github.com/Pehlevan-Group/representation-drift)

- Libsvm (version 3.22; https://www.csie.ntu.edu.tw/~cjlin/libsvm/)

- Chronux (version 2.12; http://chronux.org/) 

  For visualization:
- MatPlotLib (version 2.1.3; https://www.mathworks.com/matlabcentral/fileexchange/62729-matplotlib-perceptually-uniform-colormaps)

- slanCM (version 1.1.0; https://www.mathworks.com/matlabcentral/fileexchange/120088-200-colormap) 

Data from animals included: PPP4, PPP7, PPP8, PPP20, PPP21, PPP23, PVR1, PVR4, PVR5, PVR6

These codes were originally created in the MATLAB 2023b. 


FILES and FOLDERS
=================
  ./Figure1   : subfolder containing scripts for Figure 1
  ./Figure2   : subfolder containing scripts for Figure 2
  ./Figure3   : subfolder containing scripts for Figure 3
  ./Figure4   : subfolder containing scripts for Figure 4  
  ./Figure5   : subfolder containing scripts for Figure 5
  ./Figure5   : subfolder containing scripts for Figure 6
  ./Figure5   : subfolder containing scripts for Figure 7
  ./src       : subfolder containing demonstrations of key analysis functions
  ./utilities : subfolder containing all the helper functions.
  ./toolbox   : subfolder containing all the toolboxes used. Please refer to the original work by the developers (see the Toolboxes section for more details).


CITING OUR WORK
===============

If you find the code useful, please cite the code source and the paper:
Tang, W.*, Chang, H.*, Liu, C., Perez-Hernandez, S., Zheng, W. Y., Park, J., ... & Fernandez-Ruiz, A. (2025). Circuit mechanisms for disentangled representations and generalization in the hippocampus. Nature Neuroscience, 2026.

OR    
Tang, W.*, Chang, H.*, Liu, C., Perez-Hernandez, S., Zheng, W. Y., Park, J., ... & Fernandez-Ruiz, A. (2025). A hippocampal population code for rapid generalization. bioRxiv, 2025-03.


CONTACT
=======
Bug reports, comments and questions are appreciated.
Please write to: 
	Wenbo Tang <wenbo.tang07@gmail.com>
