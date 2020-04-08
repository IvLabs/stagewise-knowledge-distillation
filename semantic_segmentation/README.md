# Semantic Segmentation

### Usage
- The following script
    - downloads the data (and shifts it to appropriate folder)
    - downloads the pretrained teacher weights in appropriate folder
```
# assuming you are in the root folder of the repository
cd semantic_segmentation/scripts
bash setup.sh
```

- Create virtual environment (unless you already have one)
```
# assuming you are in the root folder of the repository
# a requirements.yml file is provided for conda users
conda env create -f environment.yml
```

- Setup the packages in your environment (ensure your desired environment is already activated).
```
# following command uses the setup.py file to setup the packages
python -m pip install -e .
```