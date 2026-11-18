# Spokes Tracking

This is a code that tracks spokes that have been segmented in a binary segmentation mask.

## Introduction
Spokes are dust clouds over the rings of Saturn. They have been observed several times, and are a very interesting phenomena. For this project, we are concerned with mainly the B-ring of Saturn, and here we observe several kinds of spokes- dark against a light background, light against a dark background, mixed, and more[^1]. For our project, we are concerned with analyzing these spokes to determine certain valuable patterns and interesting statistics. In order to accomplish this, I envision our pipeline looking as follows:
<br>
```text
  Raw Images --> Segmentation Model --> Tracking Algorithm --> Statistics
```

This project deals with the tracking algorithm part of this pipeline- given a binary segmented mask of spokes, can we track these spokes across images?

The code is modular and breaks the problem down into subcomponents. All of the code can be found in [./code](./code).

## How It Works

### Motion Model

Given a *frame*, we predict the position of all spokes in that frame for the next frame. For this, we assume that spokes follow a simple Keplerian motion model. This model can be described as follows:
$$
\begin{aligned}
  \theta_{k+1} &= \theta_k + \omega_r \Delta t \\
  \text{where } \omega &= \sqrt{{GM} \over r^3}
\end{aligned}
$$
So, for every longitude and radius coordinate of a spoke, we calculate the new longitude value based on the angular velocity for that radius value, and the given time difference. <br>
For every such prediction, we propagate the track id from the previous mask, and then we "superimpose" this prediction on the new observation. 

### Cost Function

The way this works is that we have a cost metric, and for every spoke pair between the predicted frame and the observed frame, we calculate this cost as such:
$$
\begin{aligned}
  c &= {d(s_1, s_2) \over {ds}} - (10 i(s_1, s_2))^2 \\
  \text{where } ds &= 20,000
\end{aligned}
$$
$d(s_1, s_2)$ gets the distance between spokes $s_1$ and $s_2$ by calculating a simple distance between the centroids of those spokes, and $i(s_1, s_2)$ calculates the Intersection Over Union (IoU) score for both spokes. $ds$ stands for distance scale, and is a way to make distance a unit-less quantity so that it can be used for the cost function. We arbitrarily choose $ds$ to be 20,000.

### Matching Function

To propagate ids between the predicted frame and the observed frame, we use a brute force approach and compare every possible pair of spokes between the two frames. We calculate the cost for each comparison, and say that the spokes that have the least cost must be the same spoke, and we assign track ids accordingly. <br>
For each track id, we assign a unique color so that we can visualize the spokes being tracked.
