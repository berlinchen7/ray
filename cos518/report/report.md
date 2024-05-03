
## Introduction

## Ray Overview

## Pong on Ray: implementation and evaluation

<!-- Details about Pong + Ray implementation, perhaps screenshot of Pong game itself. -->



In particular, whereas much of the RL evaluation found in the original Ray paper evaluate large-scale workloads (e.g., hundreds to thousands of CPUs for the Evolution Strategies task), in our investigation we are interested in behavior of the system under conditions that are more realistic to a university laboratory (e.g., tens of cpus).

In what follows, we will use the aforementioned set-ups as case studies to illustrate some properties of Ray that we have found to be especially relevant to university researchers hoping to speed up their research progress. 

## Locality Can Still Matter

<!-- Insert picture about the graph -->

## Excessive Compute Can Increase Latency

A popular paradigm in developing machine learning models (in academia at least) involves testing a small model that doesn't not require a lot of compute, which would allow for a fast iterative testing of different configurations. Once a promising configuration is found, we then run a scaled up version with more compute resources, e.g., submitting a slurm job on a cluster. 

As a consequence, often code that is optimized for the scaled up version is used for running the smaller version (and vice versa), in order to mitigate the friction that goes into the context-switching to the scaled up version, thereby reducing bugs. So in particular, if a user uses Ray to do distributed training on the scaled up model, it is likely that they will use a similar code to test smaller models. Hence, we are curious how Ray behaves when a set-up that is meant for a big system is ran on a small system instead.

More concrete, we tests 



