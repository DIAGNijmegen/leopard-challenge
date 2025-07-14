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
<pre lang="markdown"> pip3 install -r requirements.txt  </pre>. Installation of all the libraries on a basic consumer laptop can take above 1 hour.

### Hardware requirements

The inference of Docker containers submitted by participants requires the following hardware: 1x NVIDIA T4 GPU, 16 GiB GPU memory(vRAM), 8 CPUs, 32 GiB main memory (dRAM), 1 x 225 GB NVMe SSD. The prediction time on the aforementioned hardware is 30 minutes per single slide. The evaluation scripts can be executed on a basic consumer laptop without a GPU. The runtime of the longest executing script is < 40 min, depending on the hardware available.

### LEOPARD study data 

The LEOPARD study data consists of whole slide images and follow-up data from 4 cohorts: RUMC, PLCO, IMP, and UHC. The RUMC data includes Development, Tuning, and Internal Validation. Only the RUMC Development data is released publicly under CC-BY-NC-SA license, available at: https://aws.amazon.com/marketplace/pp/prodview-2cwmi5kl3oinu . 


### Getting predictions of the algorithms 

The Docker containers are programmed to read the input whole slide image from `\input` and write the output JSON files with prediction to `\output`. The containers read one case per time.
The input and output paths should be mounted with each container in the following way, where `SLIDE_ID` is the particular case ID name.

<pre lang="markdown">
    
SLIDE_PATH=${BASE_SLIDE_PATH}/${SLIDE_ID}.tif
MASK_PATH=${BASE_MASK_PATH}/${SLIDE_ID}_tissue.tif
OUTPUT_JSON=${OUTPUT_BASE_PATH}/${SLIDE_ID}.json
    
--container-mounts=${SLIDE_PATH}:/input/images/prostatectomy-wsi/${SLIDE_ID}.tif,${MASK_PATH}:/input/images/prostatectomy-tissue-mask/${SLIDE_ID}_tissue.tif,${OUTPUT_BASE_PATH}:${OUTPUT_BASE_PATH}        
</pre>
    
The containers can be provided for reviewers upon request.

### Running evaluation code

After installing the necessary libraries, specify the correct local paths to the ground truth data, predictions in the `config-test.json`. First, run the scripts that generate tabular data, and afterwards run the scripts to generate figures. The simulated dataset includes only 5 samples. If you want to run the code that has bootstrapping or permutation, you would need to decrease the amount of resampling, as resampling the small dataset will result in no admissible pairs found. 
