# C3D-EgoGesture
C3D model for EgoGesture video recognition
## Requirements
torch == 1.2 \
torchvision == 0.4.0 \
scipy \
sklearn \
tqdm \
torchsummary \
pillow==6.2.1 \
matplotlib \
opencv-python-headless \
pandas \
scikit-image
## Usage 
First use create_annotation.py to create annotation files. \
Then run C3D_fine_tune.py or R3D_fine_tune_all.py to train C3D or R3D model.

## Pretrained model
The C3D pretrained model can be download on this [link](https://github.com/DavideA/c3d-pytorch). \
The R3D pretrained model is [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M). The originial repos is on this [link](https://github.com/kenshohara/3D-ResNets-PyTorch).


