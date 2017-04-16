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
Another very good reason to work in Python is that, as you probably know, deep learning is getting state of the art results in a lot of fields, one of those being medical image analysis. During my internship I have seen deep learning beating previous methods in a lot of tasks including segmentation, super resolution, diagnosis, landmark detection,...
In this blog I will talk about the libraries that I have used for deep learning, and how I have used them in my research. In particular, I will talk a little about Caffe, Tensorflow and Pytorch.


![_config.yml]({{ site.baseurl }}/images/config.png)

The easiest way to make your first post is to edit this one. Go into /_posts/ and update the Hello World markdown file. For more instructions head over to the [Jekyll Now repository](https://github.com/barryclark/jekyll-now) on GitHub.
