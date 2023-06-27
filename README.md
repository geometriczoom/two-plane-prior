# Learned Two-Plane Perspective Prior based Image Resampling for Efficient Object Detection

Minimal Implementation of CVPR 2023 paper _Learned Two-Plane Perspective Prior based Image Resampling for Efficient Object Detection_ 
[[paper]](https://arxiv.org/abs/2303.14311) [[website]](https://geometriczoom.github.io/) [[video]](https://www.youtube.com/watch?v=gUcC0JU1bmg).

## Steps

Download Argoverse-HD from official website [here](http://www.cs.cmu.edu/~mengtial/proj/streaming/).

Code implementation uses Python 3.8.5, PyTorch 1.6.0, and mmdetection 2.20.0 and kornia 0.5.11. To set
up the conda environment used to run our experiments, please follow these steps from some initial directory:

1. Create the conda virtual environment and install packaged dependencies. You should install [miniconda](https://docs.conda.io/en/latest/miniconda.html) if not already installed.
   ```
   conda create -n tpp python=3.8.5 && conda activate tpp
   conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
   pip3 install tqdm html4vision scipy
   ```
2. Install mmdetection 2.20.0. This will first require installing mmcv 1.3.17.
   ```
   pip3 install openmim
   mim install mmcv-full==1.3.17
   mim install mmdet==2.20.0
   pip3 install mmpycocotools
   pip3 install kornia==0.5.11
   ```
3. Install tpp
   ```
   git clone https://github.com/geometriczoom/two-plane-prior.git && cd two-plane-prior
   pip install . && cd ..
   ```
4. Download checkpoint from [Google Drive](https://drive.google.com/file/d/1sT3zpjp4fV62tLJ719epEKDjDzn0Ky0d/view?usp=sharing).
   Your final directory structure should look something like this:
   ```
   data/Argoverse/
      ├── Argoverse-1.1/
      └── Argoverse-HD/

   checkpoints/
   └── KDE_TPP.pth
   ```
5. Run script to evaluate on Argoverse-HD.
   ```
   sh experiments/KDE_TPP.sh
   ```

## Citations and Credits

If you use this code, please cite:

```
@inproceedings{ghosh2023learned,
  title={Learned Two-Plane Perspective Prior based Image Resampling for Efficient Object Detection},
  author={Ghosh, Anurag and Reddy, N Dinesh and Mertz, Christoph and Narasimhan, Srinivasa G},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13364--13373},
  year={2023}
}
```

Parts of this codebase are based on [Fovea](https://github.com/tchittesh/fovea), ICCV 2021. 