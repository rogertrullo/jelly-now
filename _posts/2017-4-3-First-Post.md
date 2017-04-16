---
layout: post
title: First Post!
published: true
---

In this first post, I would like to help other people who is just starting to work with medical images.  Medical images are quite different from natural images; for example, the latter is represented as 8 bit unsigned integers, giving an intensity range of 0-255. In medical Images however, it is not uncommon to have 16 bit signed images which means that we have a wider intensity range with possibly negative numbers! You must keep this into account whenever you want to normalize or do some operations with these images. Another difference is that most of the time you will be dealing with 3D images instead of 2D, although you can thing of 3D images as a set of several 2D images which are known as slices. This figure illustrates the concept:

![alt text](http://www.cabiatl.com/mricro/mricro/batch.gif)

You will find different formats for medical images; with the most known being [DICOM](https://en.wikipedia.org/wiki/DICOM), [NIFTI](https://nifti.nimh.nih.gov/).
DICOM for example is a series of several slice images, while NIFTI only uses one snigle file. Personally I prefer the later since I only have to deal with a single file instead of one folder with several files.
Now let's say you have one of these images, how do you read it?
I remember that before I started my PhD I used to work a lot with Matlab, and then I just wanted to keep it that way. So I started to look for ways to read this images. I soon found out that most of the libraries out there were not standard and done by third party users. Although this really comes to preference, I would advise to work with Python instead. There are a couple of reasons for that; first, Python is a free open source language with a huge community online which is growing more than ever. Second, in medical images you will find that there are two main libraries that are all over the place: [ITK](https://itk.org/) and [VTK](http://www.vtk.org/). Both libraries are written in C++ and are designed mainly to work with this language; however, if you are working on research you would probably want to use a scripting language. If that's the case then you're lucky because both libraries have wrappers for python!. In [this](https://pyscience.wordpress.com/) blog you can find some really good tutorials that should help you to start going!
Another very good reason to work in Python is that, as you probably know, deep learning is getting state of the art results in a lot of fields, one of those being medical image analysis. Most of the packages used for deep learning are written for python.
During my internship I have seen deep learning beating previous methods in a lot of tasks including segmentation, super resolution, diagnosis, landmark detection,...
In this blog I will talk about the libraries that I have used for deep learning, and how I have used them in my research. In particular, I will talk a little about Caffe, Tensorflow and Pytorch.
Talking about things you'd need for your research in medical imaging is a visualization software. Most people use [ITKSNAP](http://www.itksnap.org/pmwiki/pmwiki.php) or [3DSlicer](https://www.slicer.org/). Personally I use ITKSNAP although it doesn't have all the features of 3DSlicer, it is simpler and does the job. You can visualize most of the formats, having a 3d view of the image, you can overlap segmentation, etc. Here's a pic of ITKSNAP, with a CT image that I use for my research:

![alt text](https://raw.githubusercontent.com/rogertrullo/rogertrullo.github.io/master/images/itksnap_ct.png)

So to wrap up, you'd basically need three things:
-Python
-ITKSNAP
-Images

Now, in Python, you are going to need a famous package that will allow you to do a lot of vectorized operations just as in matlab. It is called [Numpy](http://www.numpy.org/), and with it, you can use all the operations that you can do in matlab like sliceing arrays, linear algebra operations, etc. [Here](https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html) you can find a document that will help you with numpy if you come from a matlab background.
Why is this important? well, these medical images are represented as 3D arrays, and with nnumpy you can easily manipulate them, and do any kind of operation you want. To do that, first we use the ITK wrapper for Python which is called [SimpleITK](https://itk.org/Wiki/SimpleITK/GettingStarted). Here's a code you'd use for reading an image and storing it in a 3d numy array:
```python
import SimpleITK as sitk
import numpy as np

itkimg=sitk.ReadImage("path/to/image")
npimg=sitk.GetArrayFromImage(itkimage)
```

first line just reads the image into an  itkimage object. These objects have several operations that are listed in the library website including image proessing, registration, etc. Here I just need the data, and since later will be working on that, I convert it to a numpy array with the second line od code. Pretty easy, isn't it?
Now that you have the numpy array, you can do all sort of operations, like multiplying by a number, or by another array, adding or substracting, whatever you need!
For example,
```python
npimg=3*(npimg*npimg)
ctitk_mod=sitk.GetImageFromArray(npimg)
sitk.WriteImage(ctitk_mod,"path/to/save/image")
```
Would square the image and multiply it by 3. Then I convert it to an ITK image and save the file. Remember when saving the image, ou have to give the full path including the extension. For  example /folder/myimage.nii if you want Nifti.

This is it for the first post, I hope that you have at least a basic idea on how to start, or where to look for more information. In next posts, I will try to give more details taking as example my own research, where I apply deep learning to do segmentation and image synthesis.

