## Learning Biochemical Prostate Cancer Recurrence from Histopathology Slides: the LEOPARD Challenge
This repository includes code for the LEOPARD study results analysis.



### Repository structure
The repository includes evaluation code to generate statistics, tables, and figures. The `config` folder contains configuration files to run the code: `config.json` on study data, `config-test.json` on simulated data. The `demo-data` folder includes simulated data to test the code. 
<pre lang="markdown"> 
├───leopard-challenge
    ├───config
    ├───demo-data
    │   ├───clinical-variables
    │   ├───ground-truth
    │   └───predictions
    │       ├───airamatrix_1
    │       │   └───radboud
    │       ├───hitszlab_2
    │       │   └───radboud
    │       ├───martellab_2
    │       │   └───radboud
    │       ├───mevis_updated
    │       │   └───radboud
    │       └───paicon_2
    │           └───radboud
    └───src
        ├───evaluation
        └───unit_tests
        
 </pre>
 
### Installing libraries
Run the code below to install the correct versions of libraries:
<pre lang="markdown"> pip3 install -r requirements.txt  </pre>

### Hardware requirements

The inference of Docker containers submitted by participants requires the following hardware: 1x NVIDIA T4 GPU, 16 GiB GPU memory(vRAM), 8 CPUs, 32 GiB main memory (dRAM), 1 x 225 GB NVMe SSD. The prediction time on the aforementioned hardware is 30 minutes per single slide. The evaluation scripts can be executed on a basic consumer laptop without a GPU. The runtime of the longest executing script is < 40 min, depending on the hardware available.

### Getting predictions of the algorithms 

The Docker containers are programmed to read input whole slide images from `\input` and write the output JSON files with predictions to `\output`. 


### Running evaluation code

After installing the necessary libraries, specify the correct local paths to the ground truth data, predictions in the `config-test.json`. First, run the scripts that generate tabular data, and afterwards run the scripts to generate figures. 
