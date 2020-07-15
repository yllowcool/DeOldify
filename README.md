
# IBM SG Project - Breathing new life into iconic moments #

**All references** are to credit the **original author** of [DeOldify - Jason Antic](https://github.com/jantic/DeOldify), and modification of code to implement [**IBM Large Model Support (LMS)**](https://www.ibm.com/support/knowledgecenter/SS5SF7_1.7.0/navigation/wmlce_getstarted_pytorch.html#wmlce_getstarted_pytorch__lms_section), using AC922 goes to IBM ANZ team - [Ben Swinney](https://github.com/benswinney/DeOldify/tree/master/deoldify) & Adam Makarucha et al. for their published work - https://www.ibm.com/blogs/ibm-anz/breathing-new-life-into-iconic-moments/
## PRELIMINARY MODEL (16-20 June) ##
## Notes for training of colourizer model- (ColorizeTrainingStableLargeBatch.ipynb) ##
**Inspiration** of model training using **SG local images** instead of the usual IMAGENET dataset to generate output colours as close as possible to our local sights and environments; came from [ColouriseSG](https://colourise.sg/) done by SG GovTech team [last year early February 2019](https://blog.data.gov.sg/bringing-black-and-white-photos-to-life-using-colourise-sg-435ae5cc5036) (click to read their blog).
- Prototype training done with 425 training images downloaded off from **Flickr** site publicly available photos (allowed download with Creative Commons license only) 
- Quick download from Flickr made easier using [Flickr Fast Downloader tool](http://flickrdownloader.laboone.net/). Searching for "singapore" terms to cover representation of streets and people of Singapore, nature environment context, building structures and colours etc. 
- Training dataset split (75% train 25% validation)- train folder: 326 images, val folder: 99 images. 
- Used **POWER8 machine S822lc**, Nvidia GPU P100 16GB RAM, total CPU RAM at 512GB, no direct server connection, using IBM VPN, ssh remote
- **LMS training enabled** when installed FastAI using pip install (need spaCy conda install package first) into the WML-CE 1.7.0 conda environment
-  **Problems**: met with CUDA out-of-memory issue (OOM) since its small ram of 16GB in 822's P100 GPU, only able to **train up to 192px** compare to **ANZ team higher resolution of 512px**. Resnet152 (thanks to Ben for the new code) feature turned off to prevent bugs.
- tweaking of LMS parameters to prevent OOM : max_gpu_mem = 14 , halved from 29 gb to accommodate the smaller RAM 
- tweaking of batch sizes start small, epochs number to trial and error till final train & val loss at ~2.1 at Pre-GAN
- GANsavecallback iteration number tweaked to smaller at 50 to allowing saving of GAN model checkpoint
- final GAN training done with **1 repeatable cycle** : train loss 0.78, val loss 0.62 
- Total training time using the tweaked parameters and above dataset of 425 images on S822 machine: **1hour 10mins**. Setting up conda env and tweaking of parameters : ~6-10 hours. 


## Notes for inferencing- (ImageColorizerStableTests.ipynb) ##
- obtained black & white images from [PictureSG (NLB archive)](https://eresources.nlb.gov.sg/pictures)
- loaded in the saved GAN model checkpoint at cycle 1. 
- Tested B&W image include - old streets of SG ok, old NDP parades - still room for improvement, portraits of famous personalities in SG - not so good compared to the former two -> **can be solved by having a larger, more representative training dataset**
- render factor(rf) plays a part in the final output of the predicted image too (also limited by GPU RAM), 
- too high value of rf also causes glitches, colour spills, streaks

## Preliminary Results ##
A picture of Amber Road in 1992![Compare_Results_AmberRoad_rf45](https://user-images.githubusercontent.com/52671563/85354369-fe758000-b53c-11ea-8035-0659c183781f.png) 
Opening ceremony of Queenstown Branch Library on 30 April 1970, by the First Prime Minister of Singapore, Mr Lee Kuan Yew![Compare_Results_LKY1970](https://user-images.githubusercontent.com/52671563/85354420-1816c780-b53d-11ea-9171-5d1636da50c2.png)
Mr Lee Kuan Yew, Prime Minister, arriving at the Library![Compare_Results_LKY1970_1](https://user-images.githubusercontent.com/52671563/85354424-19e08b00-b53d-11ea-81d0-76b2eb748ef9.png)
National Day Parade in 1969 held at the Padang![Compare_Results_NationalDayParade1969_1](https://user-images.githubusercontent.com/52671563/85354429-1f3dd580-b53d-11ea-848e-a50627013d94.png)
Mobile Column in National Day Parade 1969 held at the Padang![Compare_Results_NationalDayParade1969_rf32](https://user-images.githubusercontent.com/52671563/85355786-f9fe9680-b53f-11ea-994d-273e9e17deb7.png)
1983 photograph showing the exterior of a house, located at no. 51 Neil Road (opposite Kreta Ayer Road)![Compare_Results_NeilRoad_rf10](https://user-images.githubusercontent.com/52671563/85354500-498f9300-b53d-11ea-8159-d572a302c691.png)
Mr S. Rajaratname, Minster of Culture, giving a speech at the opening of an exhibition at the library![Compare_Results_S_Rajaratnam1960](https://user-images.githubusercontent.com/52671563/85354504-4d231a00-b53d-11ea-9431-709072ca9fa8.png)
1982 photograph showing Maghain Aboth Synagogue, located at Waterloo Street![Compare_Results_WaterlooStreet_rf64](https://user-images.githubusercontent.com/52671563/85354509-501e0a80-b53d-11ea-834c-feb035b6a528.png)
Mr. Yusof Ishak, Head of State, at the library, est. 1950-1960![Compare_Results_YusofIshak1950](https://user-images.githubusercontent.com/52671563/85354513-52806480-b53d-11ea-86d9-7176686ce7c7.png)
*Only a small sample of images are used for demo purposes only © All Rights Reserved to NLB DIGITAL LIBRARY*

## SECOND DEVTEST RUN (15 July) ##
- Done on bigger machine POWER9 AC922, making use of larger GPU RAM of 32GB with 1x NVIDIA Tesla V100 GPU 32GB RAM only
- Imageset increased to 1000 images (750 train, 250 val)
- Training using ResNet512 backend, more depth, better perfomance results as compared to ResNet101 
- Training resolution increased up to 512px without OOM crashing out
### Conda Installation :
#### Make sure that IBM repo for WML-CE packages are in the conda channel priority list >> ####  
$ conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/  
$ conda create -n colorproj python=3.6  
$ conda activate colorproj
#### Activate the specially defined environment and do the conda install all together at once for Fastai, Pytorch, Torchvision to prevent version conflicts and to have Pytorch-LMS enabled #### 
$ conda install -c fastai -c powerai fastai pytorch torchvision // if can't install set -> conda config --set channel_priority false
#### Tensorflow is needed by Tensorboard in the training code ####
$ conda install tensorflow tensorflow-gpu
#### You can choose to use JupyterLab (preferred for better workflow) or Jupyter Notebook is the same ####
$ conda install -c conda-forge jupyterlab nodejs
#### Install jupyterlab extension widgets required for download bar indicator (only required for JupyterLab). ####
$ jupyter labextension install @jupyter-widgets/jupyterlab-manager
### README before inferencing (ImageColorizerStableTests.ipynb) : ###
1. When running inferencing 'colourizer' notebook, there will be an "no such dummy path" error when calling the vis function - thus you need to point to an image directory to circumvent that i.e. edit the file in **~/DeOldify/deoldify/dataset.py**: Line 45: path = Path('/home/cecuser/imageset/train')
2. If saved model used is generated from ResNet152 backend, then there is a need to point to the ResNet152 script in visualize.py. Change the file in **~/DeOldify/deoldify/visualize.py** : Line 7: from .generatorsResNet152 import gen_inference_deep, gen_inference_wide // or any other respective model i.e generatorsEFFNET if using EfficientNet backend
3. If colorizer is to be performed only on images and no videos are involved, you can comment out the lines that import 'ffmpeg' and 'youtube_dl' to save installation time: **~/DeOldify/deoldify/visualize.py** : Line 11: # import ffmpeg and Line 12: # import youtube_dl
4. You can remove the watermark function in **visualize.py line 31** to get rid of the small palette image created by the original author.
