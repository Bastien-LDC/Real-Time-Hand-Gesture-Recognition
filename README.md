# **Illinois Institute of Technology - CS512 - Computer Vision**
## **Final Project - Group 07 - Real-time Hand Gesture Recognition**
_Spring 2023_
---

### **📑Instructions**

#### **📂Set-up the environment**
Run the following command to install the required environment and packages:
```bash
python -m venv <env_name>
source <env_name>/bin/activate
```
or with `conda`:
```bash
conda create -n <env_name> python=3.9
conda activate <env_name>
```
Then, install the required packages:
```bash
pip install -r requirements.txt
```


#### **📂Download the dataset**

After pulling the repository, you will need to download the dataset and the weights of the model.
Download the dataset from the OneDrive link below and extract it in the `data` folder:
* [OneDrive - data](https://iit0-my.sharepoint.com/personal/bleduc_hawk_iit_edu/_layouts/15/guestaccess.aspx?guestaccesstoken=nghfcV5r5fzy4f5Y1FbyFZky48p%2BTjz%2Bp0iLDErKq8A%3D&folderid=2_0515eeabcfb45419690e62cbc9d8293ef&rev=1&e=RGzNMW)

⚠ Don't forget to change the path in the notebook to the path of the extracted dataset.

Download the weights of the model from the OneDrive link below and extract it in the `models` folder:
* [OneDrive - models](https://iit0-my.sharepoint.com/personal/bleduc_hawk_iit_edu/_layouts/15/guestaccess.aspx?guestaccesstoken=1xiX%2BioK1eNgLQi3NqNyyovwoDciSy%2Fx4LclM7qSkBg%3D&folderid=2_00147942c86a1407f906be2f0864f0eaf&rev=1&e=JJJY68)

***Note 1📝:*** _Must use IIT account for viewing access. Otherwise, the original dataset can be downloaded [here](https://github.com/hukenovs/hagrid#downloads)._


#### **▶ Run the script**
After having replaced the path of the dataset in the notebook as well as the path of the model weights, you can run either of the 2 `main.py` or `main2.py` scripts to run the program. The first one will run the program with the MediaPipe model framework (bbox and landmarks) + gesture recognition model, while the second one will run the program with our Custom model (bbox and class label).

***Note 2📝:*** _If running `main2.py`, be mindful of loading the desired models inside the script and adjusting the flags value conveniently (CLASSIFIER: True if loading a model that predicts (bboxes, class); SQUARE_IMG: True if using any MobileNet model)._

#### **🎥🖐Demo**
You can find 2 demonstration videos of our models. This [first one](https://drive.google.com/file/d/1KAfz3nFLKgtjNB3W0xnKQqzu7ScWX8o2/view?usp=share_link) shows the top model of the pipeline (MediaPipe framework). This [second one](https://drive.google.com/file/d/10JPJ1c-6zgAFrHZJmYYDBtyVmEsRrb7h/view?usp=share_link) shows the bottom model of the pipeline (Custom model).

### 👥**Team Members**

-   Bastien LEDUC
    -   A#: A20520860
    -   Email: bleduc@hawk.iit.edu
-   Paul LEGOUT
    -   A#: A20522029
    -   Email: plegout@hawk.iit.edu
