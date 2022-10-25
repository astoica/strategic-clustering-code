# strategic-clustering-sagt22

This repository contains all the code used for the "Strategic clustering" paper:
- The datasets folder contains the three datasets used for this submission: APS, Facebook, and Highschool datasets.
- The following files contain the code used in the experimental analysis of Sections 4, 5, and 6 (all written for Python 3): 
  - section4-nashequilibria-realdatasets.py contains the code for running experiments from Section 4 and plotting Figures 2 and 3.
  - section5-tradeoffalgorithm.py contains the implemented algorithms described in Section 5; Figure 4 plotted using these algorithms ran for a network created by a Stochastic Block Model with two communities 
  - section6-statparity-realdatasets.py contains the code for running experiments from Section 6 and plotting Figures 5 and 6.

For running any of the above code, uncomment the section that relates to the data desired to use (APS, Highschool, Facebook).

Note: newest version of the trade-off algorithm is in strategic-clustering-tradeoffalg-cleanedup.ipynb.
