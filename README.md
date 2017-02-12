Thesis

This program takes as input a .caffemodel and a .prototxt files to configure a CNN to perform depth estimation.
It is possible to use any desired CNN, but three inputs are already pre-set. These models are described in http://sira.diei.unipg.it/supplementary/ral2016/extra.html and should be downloaded from here. 

The following directories structure should be used:
->Home
   -build
   -mix_eigen ->save proto and caffemodel from eigen
   -mix_fcn -> save proto and caffemodel from fcn
   -mix_lstm -> save proto and caffemodel from lstm
   -images -> save images in default location
   
How to run the program:

->./merge 
The user will be asked to choose between one of the three pre-set models and pick an image

-> ./merge _path_to_proto _path_to_caffemodel _path_to_image

The output dependes on the chosen CNN and the final depth map is saved in ../images
All paths are relative to the build directory
 
