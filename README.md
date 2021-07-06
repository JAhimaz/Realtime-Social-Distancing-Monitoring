# Realtime-Social-Distancing-Monitoring

Realtime social distance monitoring system utilising Darknet YOLOv4 and OpenCV

<hr>

### Setup

1) Following this tutorial to setup the Prerequisities of OpenCV and Darknet.
(Ofcourse, a anaconda environment can be used as well if implementation is known)

[YOLOv4 Tutorial #1 - Prerequisites for YOLOv4 Installation in 10 Steps](https://www.youtube.com/watch?v=5pYh1rFnNZs)

2) Once complete, place `sd_app.py` in the `<FILE PATH>\darknet\build\darknet\x64`
3) Open `sd_app.py` in the IDE of your choice, and edit line `338` to the pre-recorded input or set line `332` to True for realtime capture.

```py
    #=============================================================================
    # TRUE = Realtime live video capture (Requires Camera) 
    # FALSE = Analyse a video. Requires Input to Video Source
    #=============================================================================
    realtimeCapture = False
    #=============================================================================

    if realtimeCapture:
        cap = cv2.VideoCapture(0)
    else:    
        cap = cv2.VideoCapture("./Input/test4.mp4")
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    if realtimeCapture:
        new_height, new_width = frame_height, frame_width
    else:    
        new_height, new_width = frame_height // 2, frame_width // 2
```


4) Run `python sd_app.py`
5) Select `4 points` of the frame for the perspective warp in the following order (Bottom Left, Bottom Right, Top Left, Top Right)
6) Select an additional `2 points` of a reference object that is 2 meters, these 2 points have to be parallel to the bottom left and bottom right points.
7) One more click anywhere on the screen shall prompt the detection scenario.
8) Output can be changed in line `352` and `357` for the perspective and birds-eye view respectively 

<hr>

### Example Output

