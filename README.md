
![Alt text](demo/ssim-score-gui-demo.gif)

Simple GUI application to check visual quality of an image with the help of SSIM, DSSIM and PSNR. 

- Drag and drop an image onto the window to check its quality. 
- Ability to quickly view SSIM, DSSIM, and PSNR-AVG metrics.
- Use the slider to auto-adjust and recalculate the metrics.
- Input desired PSNR or DSSIM value and application will automatically adjust the quality of an image ["Optimize"].
- Ability to export the optimized image by clicking on the “Export Image” button and selecting a location to save the image.

# Setup

1. Setup local venv environment ([VScode way](https://code.visualstudio.com/docs/python/environments), or [manually](https://www.youtube.com/watch?v=28eLP22SMTA&)
)
    - Open terminal in folder where you want to keep your project
    - type `code .` to open folder as the new project in VScode
    - `CMD + SHIFT + P` -> "Python: Create Environment" -> "venv" (you might need to install Python extension)
    - `CMD + SHIFT + P` -> "Python: Create Terminal" - this should start a new Virutal Environment

2. Clone the repo
3. Install the required dependencies:

```shell
pip install -r requirements.txt # OR ->-> 
pip install PyQt5 scikit-image Pillow mozjpeg-lossless-optimization numpy
```

3. Run `main.py` either in VScode or as `python3 ./main.py`


