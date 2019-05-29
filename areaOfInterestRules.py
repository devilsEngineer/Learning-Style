#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:21:48 2019

@author: ananthu
"""

"""
Enginering                        -> Auditory And Kinesthetic
Computer Engineering students     -> Auditory , Kinesthetic 
Manufacturing Engineering         -> Auditory 
Mechatronics Engineering students -> Auditory 
Medical                           -> Visual
"""

from pyknow import KnowledgeEngine,Fact,Rule
class learningCombination(Fact):
    """Info about the learning combination."""
    pass


class areaOfInterest(KnowledgeEngine):
    value="Other"
    @Rule(learningCombination(['A','K','V']))
    def All(self):
        self.value="Medical,Computer Engineering,Manufacturing Engineering,Mechatronics Engineering "
        
    @Rule(learningCombination(['A','K']))
    def AK(self):
        self.value="Computer Engineering,Manufacturing Engineering,Mechatronics Engineering"

    @Rule(learningCombination(['K','V']))
    def KV(self):
        self.value="Computer Engineering And Medical"

    @Rule(learningCombination(['A','V']))
    def AV(self):
        self.value="Medical,Computer Engineering,Manufacturing Engineering,Mechatronics Engineering"
        
    @Rule(learningCombination(['A']))
    def A(self):
        self.value = "Computer Engineering,Manufacturing Engineering,,Mechatronics Engineering"
        
    @Rule(learningCombination(['K']))
    def K(self):
        self.value="Computer Engineering"
        
    @Rule(learningCombination(['V']))
    def V(self):
        self.value="Medical"


