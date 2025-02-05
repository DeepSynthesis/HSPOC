# HSPOC

 H-SPOC descriptor generator

## About The Project

pKa is one of the most fundamental physicochemical properties of compounds. Microscopic pKa at specific sites are important in the researches of organic chemical reactivity, protein docking and drug design. However, the determination of micro-pKa was challenging on both experimental measurements and theoretical calculations. Although micro-pKa was a valuable concept, even in the present era, the data of micro-pKa was still lacking. The methodologies employed for the accurate predictions of micro-pKa were still developing. In this work, based on a reliable and accurate experimental pKa database: iBonD, we developed a high-precision machine learning prediction method for pKa prediction at any local sites in small molecules based on H-SPOC descriptor. The model could obtain R2 = 0.95, RMSE = 1.45 and reached the state-of-art in SAMPL6 and SAMPL7 challenges. In more testing, H-SPOC served its ability for micro- pKa prediction and conform-specific prediction.

## Getting Started

### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/DeepSynthesis/HSPOC-version-1.0
   ```

2. Create new python environment

   ```sh
   conda create -n HSPOC-env python=3.11 -y
   conda activate HSPOC-env
   pip install rdkit==2023.03.3 
   pip install networkx==3.3
   conda install pandas=2.0
   conda install numpy=1.24
   conda install scikit-learn=1.2.2
   conda install xgboost=1.7
   conda install lightgbm=4.3.0
   conda install catboost=1.2.3
   conda install matplotlib=3.7
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

1. Descriptor generation and pKa prediction:
   open `./script/hspoc_NoStructure_v1.py` ,then you can find codes as below around line 143.

   ```sh
    datafilename='csvFileName'
   ```

   change line 143 `'csvFileName'` to your datafile name. The data (.csv) must include column named `"ID"` `"solvent"` `"SMILES"` `"H_index"` `"filetype"` .

   ```sh
   cd scripts
   python hspoc_NoStructure_v1.py
   ```

   After that, the descriptors of your data will be saved in `'PredicPSPOC.csv'`, and the result of your data will be saved in `'./Pred/After******.csv'`.

2. Modeling:  

   ```sh
   python Methods.py
   ```

3. Predict contributions in pH range 0~14:

    prepare a file like `./Pred/Gly_states.csv`, the SMILES of the acid species should be ordered as the dissociation order. (column `pKa` was not necessary)

    then run

    ```sh
    python Get_pH_contribution.py 
    ```

    After prediction, you could find the results at `'./Pred/'` and a plot will be saved as `'./script/Gly_Pred.png'`



<p align="right">(<a href="#readme-top">back to top</a>)</p>
