# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 18:53:39 2018

@author: mudit
"""
aList = [1,5,6,3, 5, 4, 2, 3, 4, 3 , -1, -10, 0] 

def selection_sort(List):
    for i in range(len(List)):
        min = i
        for k in range(i,len(List)):
            if List[k] < List[min]:
                min = k
        swap(List, min, i)
    print(List)

def swap(List, x, y):
    temp = List[x]
    List[x] = List[y]
    List[y] = temp

selection_sort(aList)