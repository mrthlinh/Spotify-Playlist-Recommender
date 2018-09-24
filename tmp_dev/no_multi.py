#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 01:42:02 2018

@author: bking
"""

from multiprocessing import Pool
import time

def sum_prime(num):
    
    sum_of_primes = 0

    ix = 2

    while ix <= num:
        if is_prime(ix):
            sum_of_primes += ix
        ix += 1

    return sum_of_primes

def is_prime(num):
    if num <= 1:
        return False
    elif num <= 3:
        return True
    elif num%2 == 0 or num%3 == 0:
        return False
    i = 5
    while i*i <= num:
        if num%i == 0 or num%(i+2) == 0:
            return False
        i += 6
    return True

if __name__ == '__main__':
    start = time.time()
    r = list(range(1000000, 7000000, 1000000))
    for i in r:
        print(sum_prime(i))
    print("Time taken = {0:.5f}".format(time.time() - start))