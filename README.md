# FASText

### FASText: Efficient Unconstrained Scene Text Detector,Busta M., Neumann L., Matas J.:  ICCV 2015.  
  - http://cmp.felk.cvut.cz/~neumalu1/neumann_iccv2015.pdf

To build a standalone library, run
```
mkdir Release
cd Release
cmake -D CMAKE_BUILD_TYPE=Release ..
make 
```
Prerequisites:
  - OpenCV 
  - python + numpy (optional)

After building the executables, you can use toy examples in python:
```
cd tools
python segmentation.py <path_to_image>
```  
  - will process and draw FASText keypoints on scale pyramid. 

```
cd tools
python evaluateSegmentation.py
```  
  - will reproduce results on ICDAR 2013 dataset (requires Challenge 2 dataset & GT segmentations) 

For text lines clustering, you can run the simple demo: 
```
cd tools
python testLines.py
```
The text line clustering is just simple voting in Hough space where each region vote for each line going through region centroid

Please cite this paper if you use this data or code:
```
@InProceedings{Busta_2015_ICCV,
  author = {Busta, Michal and Neumann, Lukas and Matas, Jiri},
  title = {FASText: Efficient Unconstrained Scene Text Detector},
  journal = {The IEEE International Conference on Computer Vision (ICCV)},
  month = {June},
  year = {2015}
}
```
