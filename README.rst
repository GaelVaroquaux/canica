
======================================
CanICA: Canonical ICA for fMRI data
======================================

What is CanICA?
================

CanICA is an ICA package for group-level analysis of fMRI data. Compared
to other packages, it brings a well-controlled group model, as well as a
thresholding algorithm controlling for specificity and sensitivity with
an explicit model of the signal. The reference papers are:

    * G. Varoquaux et al. "A group model for stable multi-subject ICA on
      fMRI datasets", NeuroImage Vol 51 (2010), p. 288-299 

    * G. Varoquaux et al. "ICA-based sparse features recovery from fMRI
      datasets", IEEE ISBI 2010, p. 1177

Pre-prints for both papers are available on hal
(http://hal.archives-ouvertes.fr)

Using CanICA
==============

Currently, CanICA is used through it's Python programming interface. Get
started with the example Python script available in the examples
directory.

.. warning:: **Memory usage**

    If you are planing to work on multisubject data, you will most likely
    use a lot of memory. This means that you will probably need a
    **64bit** OS and version of Python. If you get a *MemoryError* when
    running CanICA on your data, you should switch to 64bit.

How CanICA compare to Melodic or Gift?
=======================================

As of Spring 2010, CanICA is more reproducible and offers better
statistical control.

On the other hand, it lacks a nice user interface. In addition, it does
not offer a way to estimate back-projected subject-level components,
similar to dual-regression. For someone with a mathematical understanding
of dual-regression, it should be easy to apply dual-regression on the
output maps of CanICA.

Installing CanICA
===================

CanICA is meant to be merged in the NiPy collaborative project for
NeuroImaging in Python soon.

Currently, to install CanICA, you need to install:

    * **Standard scientific Python packages**: recent versions of 
      numpy, scipy, matplotlib, and sympy. The easiest way to install
      these is to rely on a distribution. Either your Linux distribution,
      if you are running Linux, or EPD  or Python(x,y) -
      http://www.enthought.com/products/epd.php
      http://pythonxy.com
      
    * **NiPy**: NeuroImaging in Python 
      http://nipy.sourceforge.net/nipy/stable/index.html
      We suggest that you install the latest NeuroSpin branch of NiPy:
      http://github.com/neurospin/nipy/zipball/master

      You can check out the NiPy install instructions for more details:
      http://nipy.sourceforge.net/nipy/stable/users/installation.html

    * **joblib**: 
      http://pypi.python.org/pypi/joblib

You can download the latest CanICA code from
http://github.com/GaelVaroquaux/canica/zipball/master

.. note::

    Python packages can be installed from source using the following
    command::

	python setup.py install

Licensing
==========

CanICA is **BSD-licenced** (3 clause):

    This software is OSI Certified Open Source Software.
    OSI Certified is a certification mark of the Open Source Initiative.

    Copyright (c) 2009-2010, Gael Varoquaux
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, 
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of Gael Varoquaux. nor the names of other CanICA 
      contributors may be used to endorse or promote products derived from 
      this software without specific prior written permission.

    **This software is provided by the copyright holders and contributors
    "as is" and any express or implied warranties, including, but not
    limited to, the implied warranties of merchantability and fitness for
    a particular purpose are disclaimed. In no event shall the copyright
    owner or contributors be liable for any direct, indirect, incidental,
    special, exemplary, or consequential damages (including, but not
    limited to, procurement of substitute goods or services; loss of use,
    data, or profits; or business interruption) however caused and on any
    theory of liability, whether in contract, strict liability, or tort
    (including negligence or otherwise) arising in any way out of the use
    of this software, even if advised of the possibility of such
    damage.**

