# A Touchscreen Latency Study
## Tap and Drag Latency measuring tool(Automatic)


## Introduction

```
Latency in a system’s response to user input adversely impacts user performance.
Therefore, system designers need an approachable and effective way to measure such latency to determine its impacts.
However, currently proposed approaches require additional equipment or much manual work.
These requirements limit the usability of the methods.
A study shows that any increase in latency decreases user performance, and users may be able to perceive latency as low as 1ms (Jota et al., 2013).
An effort to accurately measure the latency will be considerably helpful for enhancing the latency performance to improve user experience.
Therefore, I propose an automated latency measuring system using an object tracking algorithm.
This system automatically measures the distance between the finger position and the tracking graphic by converting the number of pixels to physical distance.
The latency can be measured by estimating the velocity of finger movement.
This approach only requires a camera and a device which will ease much burden on users.
Using this automated system, a user can measure an accurate touchscreen latency only by clicking the target area on the screen.
```

## Used Algorithm
- Object detection
- Object tracking
- Color Detection


## Used Library
- OpenCV
