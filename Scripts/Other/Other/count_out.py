#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:19:10 2023

@author: floriancurvaia
"""
count=0
with open("out.txt", "r") as res:
    for row in res:
        line=row.strip().split("/")
        if len(line)>1 and line[1]=="data":
            count+=1
print(count)