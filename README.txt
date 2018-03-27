Pix2Pix
----

1. Using an internet browser navigate to the url "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/"
2. Download and unpack the maps.tar.gz dataset so that the maps folder is in the same folder as splitter.py
3. Run Splitter.py to fill the "input/data" and "target/data" folders with images, add the folders if needed
4. Ensure there is an empty "results" folder for output to be saved to
5. Run "Test.py", if you run the code by pasting it into a Jupyter notebook then uncomment the line %matplotlib inline*
* Issues with output have occured in certain environments. The Best way to run is code is to copy it into a Jupyter notebook

The hyperparameters are located near the top of the Test.py file






GTA dataset 
----

1. Download the original GTA maps from "http://gtaforums.com/topic/595113-high-resolution-maps-satellite-roadmap-atlas/", and save them to the folder gta_images


2. Run gta_download.py to seperate the file
3. Ensure there is an empty "results" folder for output to be saved to

4. Run "Patch_model.py" or "Pixel_model.py"
