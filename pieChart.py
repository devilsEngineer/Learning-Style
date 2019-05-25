#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 10:36:49 2019

@author: ananthu
"""

# import the pyplot library

import matplotlib.pyplot as plotter

# The slice names of a population distribution pie chart

Labels= 'Asia', 'Africa', 'Europe', 'North America', 'South America', 'Australia'

# Population data
Share  = [59.69, 16, 9.94, 7.79, 5.68, 0.54]

def drawPieChart(Labels,Share,name):
    figureObject, axesObject = plotter.subplots()

    # Draw the pie chart

    axesObject.pie(Share,labels=Labels,autopct='%1.2f',startangle=90)
 
    # Aspect ratio - equal means pie is a circle

    axesObject.axis('equal')
    plotter.savefig(name)
    plotter.show()
    