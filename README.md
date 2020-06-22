
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
- obtained black & white images from [SG Photos (NLB archive)](https://eresources.nlb.gov.sg/pictures)
- loaded in the saved GAN model checkpoint at cycle 1. 
- Tested B&W image include - old streets of SG ok, old NDP parades - still room for improvement, portraits of famous personalities in SG - not so good compared to the former two -> **can be solved by having a larger, more representative training dataset**
- render factor(rf) plays a part in the final output of the predicted image too (also limited by GPU RAM), 
- too high value of rf also causes glitches, colour spills, streaks
