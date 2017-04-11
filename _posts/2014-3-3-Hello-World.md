---
layout: post
title: First Post!
published: true
---

In this first post, I would like to help other people who is just strating to work with medical images.  Medical images are quite different from natural images; for example, the latter is represented as 8 bit unsigned integers, giving an intensity range of 0-255. In medical Images however, it is not uncommon to have 16 bit signed images which means that we have a wider intensity range with possibly negative numbers! You must keep this into account whenever you want to normilize or do some operations with these images. Another difference is that most of the time you will be dealing with 3D images instead of 2D, although you can thing of 3D images as a set of several 2D images which are known as slices. This figure illustrates the concept:
![]({{site.baseurl}}/http://www.cabiatl.com/mricro/mricro/batch.gif)

You will find different formats for medical images; with the most known being [DICOM](https://en.wikipedia.org/wiki/DICOM), [NIFFTI](https://nifti.nimh.nih.gov/).



I remeber that before I started my PhD I used to work a lot with Matlab, and then I just wanted to keep it that way. So I started

![_config.yml]({{ site.baseurl }}/images/config.png)

The easiest way to make your first post is to edit this one. Go into /_posts/ and update the Hello World markdown file. For more instructions head over to the [Jekyll Now repository](https://github.com/barryclark/jekyll-now) on GitHub.
