# Official code repository accompanying the paper "Reinventing the wheel: Domain-focused biobehavioral assessment confirms a depression-like syndrome in C57BL/6 mice caused by chronic social defeat"

## Getting started

To install the required packages, run:

```bash
mamba env create -n DLS python=3.12
mamba activate DLS
pip install pipx
pipx inspall poetry
poetry install
```

All code used to reproduce the results in the paper can be found in the `notebooks` directory.

## DZMap

Together with the paper, we release the DZMap, a streamlit-based visualization tool that allows researches to both explore our data and to upload their own, to check whether their cohort is compatible with the DLS model. 

To run the DZMap locally, you can use the following command:

```bash
streamlit run DZMap/dzmap.py
```

COMING SOON: The DZMap can be found at https://dzmap.streamlit.app/.

