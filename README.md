<h1>YOLOV1: Build from scratch</h1>
<div align="center" dir="auto">
<a href="https://github.com/LuongTuanAnh163002/Yolov1s/blob/main/LICENSE"><img src="https://camo.githubusercontent.com/00b6aa098f95cc8559f5f72a62f63261e44a1f09f0f560ca4c8ab25d4a631f05/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c6963656e73652d4d49542d253343434f4c4f522533452e7376673f7374796c653d666f722d7468652d6261646765" alt="Generic badge" data-canonical-src="https://img.shields.io/badge/License-MIT-%3CCOLOR%3E.svg?style=for-the-badge" style="max-width: 100%;"></a>
<a href="https://pytorch.org/get-started/locally/" rel="nofollow"><img src="https://camo.githubusercontent.com/0add0c0b6ec6267b61016063796469feb03cc17c93d9f04201e25d0f12651de0/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5059544f5243482d312e31302b2d7265643f7374796c653d666f722d7468652d6261646765266c6f676f3d7079746f726368" alt="PyTorch - Version" data-canonical-src="https://img.shields.io/badge/PYTORCH-1.10+-red?style=for-the-badge&amp;logo=pytorch" style="max-width: 100%;"></a>
<a href="https://www.python.org/downloads/" rel="nofollow"><img src="https://camo.githubusercontent.com/c2623d41ae89703a8d56dab2e458028b95b87d8ce1897ff29930ef267e9e77e0/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f505954484f4e2d332e372b2d7265643f7374796c653d666f722d7468652d6261646765266c6f676f3d707974686f6e266c6f676f436f6c6f723d7768697465" alt="Python - Version" data-canonical-src="https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&amp;logo=python&amp;logoColor=white" style="max-width: 100%;"></a>
<br></p>
</div>

<details open="">
  <summary>Table of Contents</summary>
  <ol dir="auto">
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#project-structure">Project Structure</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li><a href="#custom-dataset">How to run repository with custom dataset</a></li>
    <li><a href="#colab">Try with example in google colab</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#about-the-project">About The Project<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>

<p dir="auto">In this project we will build YOLOV1 from scratch and training with VOC dataset</p>
<img width="100%" src="https://github.com/LuongTuanAnh163002/Yolov1s/blob/main/images/decription.jpg" style="max-width: 100%;">
<h3>Yolov1 model architech</h3>
<img width="100%" src="https://github.com/LuongTuanAnh163002/Yolov1s/blob/main/images/yolov1_architech.jpg" style="max-width: 100%;">
<h3>VOC dataset</h3>
<img width="100%" src="https://github.com/LuongTuanAnh163002/Yolov1s/blob/main/images/Voc.png" style="max-width: 100%;">
<h3>Loss function</h3>
<img width="100%" src="https://github.com/LuongTuanAnh163002/Yolov1s/blob/main/images/loss.png" style="max-width: 100%;">

