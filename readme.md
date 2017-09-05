<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#sec-1">1. 31/08/2017, THURSDAY</a>
<ul>
<li><a href="#sec-1-1">1.1. <span class="done DONE">DONE</span> Create Summary of labels.</a></li>
<li><a href="#sec-1-2">1.2. <span class="done DONE">DONE</span> Load PIL or other image library</a></li>
<li><a href="#sec-1-3">1.3. <span class="done DONE">DONE</span> Pick a model that is already fine tuned for videos</a></li>
<li><a href="#sec-1-4">1.4. <span class="todo ABORTED">ABORTED</span> In case ^ fails, I should begin looking for a simple seg-net in keras</a></li>
</ul>
</li>
<li><a href="#sec-2">2. 01/09/2017, FRIDAY</a>
<ul>
<li><a href="#sec-2-1">2.1. Review:</a></li>
<li><a href="#sec-2-2">2.2. <span class="done DONE">DONE</span> List Slim pros and Cons</a></li>
<li><a href="#sec-2-3">2.3. <span class="done DONE">DONE</span> List Keras pros and Cons</a></li>
<li><a href="#sec-2-4">2.4. <span class="done DONE">DONE</span> pick slim or Keras</a></li>
<li><a href="#sec-2-5">2.5. <span class="todo TODO">TODO</span> Train and Eval a First S***y Model</a></li>
<li><a href="#sec-2-6">2.6. <span class="todo TODO">TODO</span> TF throws warnings when running convert<sub>caffe</sub><sub>model</sub></a></li>
</ul>
</li>
<li><a href="#sec-3">3. 02/09/2017, SATURDAY</a>
<ul>
<li><a href="#sec-3-1">3.1. Review:</a></li>
<li><a href="#sec-3-2">3.2. <span class="done DONE">DONE</span> Push to github</a></li>
<li><a href="#sec-3-3">3.3. <span class="todo TODO">TODO</span> Replace top-layer(s) with fcs for target vector</a></li>
<li><a href="#sec-3-4">3.4. <span class="todo TODO">TODO</span> Train new top-layer(s) with autoencoding</a></li>
<li><a href="#sec-3-5">3.5. <span class="todo TODO">TODO</span> script AWS for training top-layer(s)</a></li>
<li><a href="#sec-3-6">3.6. <span class="todo TODO">TODO</span> Adapt cv2 code for gpu</a></li>
</ul>
</li>
</ul>
</div>
</div>



# 31/08/2017, THURSDAY<a id="sec-1" name="sec-1"></a>



## DONE Create Summary of labels.<a id="sec-1-1" name="sec-1-1"></a>

1.  Write Script locally and test. 
    -   For train01, not all the tools are used.

## DONE Load PIL or other image library<a id="sec-1-2" name="sec-1-2"></a>

1.  For now, I am going to use Keras
2.  Pick one of the Keras Models under applications:
    -   The fastest is probably the best choice
3.  Determine its input image size
4.  Reduce train01.mp4 to frames of that size
5.  Evalulate the model

## DONE Pick a model that is already fine tuned for videos<a id="sec-1-3" name="sec-1-3"></a>

Mostly finding models via ActivityNet 2017 tasks 3&4 submissions

1.  DenseNet
    -   Not for videos&#x2026; but killer for still images
    -   TF Imps:
        -   YixunaLi: NO weights, No finetuning
        -   LaurentMazare: NO weights, NO finetuning
        -   other one: No Weights&#x2026;
2.  CDC: Convolutional-De-Convolutional Networks for Precise Temporal \\
    Action Localization in Untrimmed Videos
    -   Cited by winners of 2017,
    -   presented at CVPR 2017
    -   Can only be used with Caffe, and even then:
        -   Only with C3D, Facebooks improvement on Caffe for 3D convs
        -   This will not build on my mac, as I do not have a gpu
        -   It should work on aws
3.  Single SHot Temporal Action Detection
    -   Tianwei Lin, Xu Zhao, Zheng Shou
4.  Temporal Convolution Based Action Proposal
    -   Tianwei Lin, Xu Zhao, Zheng Shou
5.  arxiv prepint:1608.00797
6.  Segment-CNN
    -   Another entry to CVPR 2016
    -   Looks like it could be built for CPU based on config
    -   Will attempt Build
    -   Completed build, attempting Demo
    -   Demo does not work right now, running make runtest to see if it was built right
    -   `install_name_tool -change @rpath/libhdf5.10.dylib ~/anaconda2/lib/libhdf5.10.dylib .build_release/tools/caffe`
    -   `install_name_tool -change @rpath/libhdf5_hl.10.dylib ~/anaconda2/lib/libhdf5_hl.10.dylib .build_release/tools/caffe`
    -   And there was a huge crash with explosion, even with rebuilding and linking of CV
        -   openMPI, Numpy, OpenBlas&#x2026;
        -   There involves opencv, so I will keep on eye on any other open cv errors
7.  <https://github.com/pengxj/action-faster-rcnn>
8.  <https://bitbucket.org/sahasuman/bmvc2016_code>
9.  <https://github.com/gkioxari/ActionTubes>
10. <https://github.com/jvgemert/apt>
11. <https://bitbucket.org/doneata/proposals>
    -   Python and Matlab
12. <http://www.cs.ucf.edu/~ytian/sdpm.html#Code>
           -Matlab
13. <https://github.com/escorciav/daps>
    -   Python, Theano, Pretrianed models
14. <https://github.com/cabaf/sparseprop>
    -   C3D and python
15. <https://github.com/wanglimin/actionness-estimation/>
    -   opencv, c++, denseflow, matlab
16. <https://github.com/syyeung/frameglimpses>
    -   Torch, end to end
17. <https://github.com/gulvarol/ltc>
    -   Torch, no weights
18. <https://github.com/yjxiong/temporal-segment-networks>
    -   Won ActivityNet 2016
    -   Caffe
19. <https://github.com/atcold/pytorch-CortexNet>
    -   Some weights
    -   pyTorch
20. <https://github.com/amandajshao/Slicing-CNN>
    -   Caffe
21. <https://github.com/yhenon/keras-frcnn>
    -   Keras
    -   Bounding Boxes
22. <https://pjreddie.com/darknet/yolo/>
    -   Needs bounding boxes&#x2026;
23. All of this lead to me to c3d-keras, which seems to work

## ABORTED In case ^ fails, I should begin looking for a simple seg-net in keras<a id="sec-1-4" name="sec-1-4"></a>

# 01/09/2017, FRIDAY<a id="sec-2" name="sec-2"></a>



## Review:<a id="sec-2-1" name="sec-2-1"></a>

-   Yesterday, I did look at many interesting models, but none of them will
    work out of the box. Most would take quite a few hours simply to set up
    a develop envoirment, so I am putting them off till later
-   Off the top of my head, I feel that my options are to use slim or Keras

## DONE List Slim pros and Cons<a id="sec-2-2" name="sec-2-2"></a>

-   PRO: I know it inside and out
-   PRO: video prediction model might work
    -   <https://github.com/tensorflow/models/tree/master/video_prediction>
-   PRO: It is definately properly installed on my local

## DONE List Keras pros and Cons<a id="sec-2-3" name="sec-2-3"></a>

-   CON: No video recognition models under Applications
-   PRO: Possibly C3D under model Zoo
    -   <https://github.com/albertomontesg/keras-model-zoo/blob/master/kerasmodelzoo/models/c3d.py>

## DONE pick slim or Keras<a id="sec-2-4" name="sec-2-4"></a>

-   FAILED: goign with Keras and c3d in model zoo
-   Going to try harvitronix
-   Thgough harvitronix, found c3d-keras by axon-research
    -   whihc is actually just one person&#x2026;
    -   Looks very promising.
    -   Switching to AWS to speed things along&#x2026;
-   WAITING FOR AWS TO LET ME HAVE A BLOODY INSTANCE

## TODO Train and Eval a First S\*\*\*y Model<a id="sec-2-5" name="sec-2-5"></a>

-   In leiu of AWS, I will try and get c3d-keras running locally
    -   DONE Download weights
    -   DONE Download labels
    -   DONE Download raw caffe model
    -   DONE run protoc &#x2013;python<sub>out</sub>=. caffe.proto
    -   DONE configure keras
    -   DONE run convert<sub>caffe</sub><sub>model</sub>
        -   NOTE convert<sub>caffe</sub><sub>model</sub> is not parralized and could be
    -   DONE download test vids
    -   WIP run test<sub>model</sub>.py
        -   cv2 is throwing errors, which is upsetting
        -   opencv is still not linked to homebrew&#x2026;
        -   but `pip install opencv-python` did the trick
-   Ok, so it works on the demo, but now I need to run with my stuff
-   Accomplished! We have output from a c3d model with a tf backend
    -   the output is for the

## TODO TF throws warnings when running convert<sub>caffe</sub><sub>model</sub><a id="sec-2-6" name="sec-2-6"></a>

-   If I am going to use c3d-keras, I should fix that.
-   I recorded the warnings in out.text

# 02/09/2017, SATURDAY<a id="sec-3" name="sec-3"></a>



## Review:<a id="sec-3-1" name="sec-3-1"></a>

Last night, I eneded with an output from the network that was a 500
long confidence vector, which is what I expected. The original
model was trained on a dataset with 500 labels. Also, the network
would like images to be 112 by 112, while the preprosccesing would
like the network to be 128 by 178. Now, the first goal should be to
adapt the top layer for my labels, which is an 18 long vector. That
would let me eval the network. Then I can look at clever
downsampling methods with convolutional layers with large strides,
pooling, etc. Those will probably be trained by autoencoders, which
I will have to review. I will also need to review papers on
downsampling. 

## DONE Push to github<a id="sec-3-2" name="sec-3-2"></a>

-   Branch name changed to pre-alpha
-   github repo intialized
-   issue with file size of weight matrix
    -   This is why chuckcho downloaded the wieghts from another site
    -   Which is what I will do.

## TODO Replace top-layer(s) with fcs for target vector<a id="sec-3-3" name="sec-3-3"></a>

## TODO Train new top-layer(s) with autoencoding<a id="sec-3-4" name="sec-3-4"></a>

## TODO script AWS for training top-layer(s)<a id="sec-3-5" name="sec-3-5"></a>

## TODO Adapt cv2 code for gpu<a id="sec-3-6" name="sec-3-6"></a>

-   Does python-cv support the openCV gpu module? NO
-   Solution:
    -   <https://stackoverflow.com/questions/42125084/accessing-opencv-cuda-functions-from-python-no-pycuda>
-   Great, more building is in my future
-   Also, will not build with cuda wrappers, so will have to build on