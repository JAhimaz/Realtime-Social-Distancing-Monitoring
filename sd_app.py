from ctypes import *
import math # For distance calculations
import random
import os
import cv2
import numpy as np
from PIL import ImageGrab
import time
import darknet
import keyboard
from datetime import datetime
from itertools import combinations


mouse_pts = []
crowd_detection_threshold = 60

def get_mouse_points(event, x, y, flags, param):

    #=============================================================================
    # Mouse Callback Function
    # -> Get Mouse Points is a callback function to get the 6 required points.
    # -> The Extra Click after is just to end the callback function and will be 
    #    ignored within usage.
    #=============================================================================
    
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(param, (x, y), 10, (0, 255, 255), 10) # Show a circle to show Users Input
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y)) 

def two_point_distance(p1, p2):

    #=============================================================================
    # Basic Distance between two points calculation
    # -> Calculating the distance between 2 points within the image.
    # -> Used for calculating points between centroids.
    #
    # RETURN: The distance
    #
    #=============================================================================
    
    dst = math.sqrt(p1**2 + p2**2) # Distance calculation.
    return dst 

def bbox_coords(x, y, w, h): 
    
    #=============================================================================
    # This function converts the coordinates into rectangle coordinates used for
    # the bounding box.
    #
    # x, y = midpoint of the bounding box
    # w, h = width and height of the bounding box
    #
    # RETURN: returns each point of the bounding box
    #
    #=============================================================================

    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def get_camera_perspective(img, src_points):

    #=============================================================================
    # This function is for obtaining the camera's perspective.
    # Used to convert coords to the bird's eye view coordinates.
    #=============================================================================

    IMAGE_H = img.shape[0]
    IMAGE_W = img.shape[1]
    src = np.float32(np.array(src_points))
    dst = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return M, M_inv

def get_socialdistance_birds_eye(p, M, scale_w, scale_h):

    #=============================================================================
    # Converts the perspective Social Distance points (Points 5 and 6 from user
    # input) into the birds eye view.
    #
    # RETURN: Birds eye view "6ft / 2 meters" distance calibration (Threshold)
    #=============================================================================

    # Takes the two points from the perspective camera
    pt1, pt2 = np.array([[[p[0][0], p[0][1]]]], dtype="float32"), np.array([[[p[1][0], p[1][1]]]], dtype="float32")

    # Warps it into birds eye view coordinates
    warped_pt1, warped_pt2 = cv2.perspectiveTransform(pt1, M)[0][0], cv2.perspectiveTransform(pt2, M)[0][0]

    # Calculates the distance both points into a single value (Social Distance threshold)
    socialDistance = two_point_distance((warped_pt1[0] - warped_pt2[0]), (warped_pt1[1] - warped_pt2[1]))

    return socialDistance

def cv_detect_persons(detections, img):

    #=============================================================================
    # This function is for detecting only persons within the scene.
    #
    # An alternative would be to write a custom dataset, but with the constraint
    # of time, this wasn't possible.
    #
    # RETURN: image with detections and the dictionary of current detections.
    #
    #=============================================================================

    # Function creates a dictionary and calls it centroid_dict
    centroid_dict = dict() 						

    # Checks if there is atleast one detection within the frame
    if len(detections) > 0:  				
        # We inialize a variable called ObjectId and set it to 0		  
        objectId = 0		
        # In this if statement, we filter all the detections for persons only						
        for detection in detections:				
            # Check for the only person name tag within the COCO config
            name_tag = detection[0]   
            if name_tag == 'person':     
                # Store the center points of the detections           
                x, y, w, h = detection[2][0],\
                            detection[2][1],\
                            detection[2][2],\
                            detection[2][3]
                # Convert from center coordinates to rectangular coordinates, We use floats to ensure the precision of the BBox       
                # Append center point of bbox for persons detected.            	
                xmin, ymin, xmax, ymax = bbox_coords(float(x), float(y), float(w), float(h))   
                # Create dictionary of tuple with 'objectId' as the index center points and bbox
                centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax) 
                # Increment
                objectId += 1

    return img, centroid_dict

def plot_points_on_bird_eye_view(img, centroid_dict, M, scale_w, scale_h, socialDistance):

    #=============================================================================
    # The Longest function of this program
    #
    # This plots the points on the birds eye view and calculates whether or not social distance
    # is being practiced.
    #
    # RETURN: Warped coords, birds eye view image and the red_zone list (To be used
    # for the perspective view).
    #
    #=============================================================================

    # Get the Frames height and width
    frame_h = img.shape[0]
    frame_w = img.shape[1]

    # Circle and view Variables (Static Variables)
    node_radius = 10
    thickness_node = 20
    solid_back_colour = (41, 41, 41)

    # Creates an array of the view
    blank_image = np.zeros(
        (int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8
    )

    # Sets the background to the solid gray colour
    blank_image[:] = solid_back_colour

    # Create a dictionary of each warped point, this allows us to set an ID for each point to be referenced later on.
    warped_pts = dict()
    # Initial ObjectID of 0, this will allow us to set an ID for each point
    objectId = 0

    # Setting the bird image as the blank image
    bird_image = blank_image

    # Loop through each id and points within the centroid dictionary (Perspective points) and convert it to
    # Birds eye view points
    for (id, p) in centroid_dict.items():
        pts = np.array([[[p[0], p[1]]]], dtype="float32")
        warped_pt = cv2.perspectiveTransform(pts, M)[0][0]
        warped_pts[objectId] = (int(warped_pt[0] * scale_w), int(warped_pt[1] * scale_h))
        objectId += 1 #Increment the index for each detection  

    # List containing which Object id is in under threshold distance condition. 
    red_zone_list = []
    red_line_list = []

    for (id1, p1), (id2, p2) in combinations(warped_pts.items(), 2):
        # Check the difference between centroid x: 0, y :1
        dx, dy = (p1[0] - p2[0]), (p1[1] - p2[1])  	
        # Calculates the Euclidean distance
        distance = two_point_distance(dx, dy)
        # Using the previously set Social Distance, use it as the threshold
        if distance < socialDistance:
            # If below the threshold, add both of the detected persons to the red_zone_list						
            if id1 not in red_zone_list:
                red_zone_list.append(id1)       #  Add Id to a list
                red_line_list.append(p1[0:2])   #  Add points to the list
            if id2 not in red_zone_list:
                red_zone_list.append(id2)		# Same for the second id 
                red_line_list.append(p2[0:2])

    # Within those points if detected below threshold, create a blue circle, else a green circle for non-red-zone persons
    for id, pts in warped_pts.items():  
        if id in red_zone_list:
            cv2.circle( blank_image,(pts[0], pts[1]),node_radius,(255, 0, 0),thickness_node )
        else:
            cv2.circle( blank_image,(pts[0], pts[1]), node_radius,(0, 255, 0),thickness_node )

    # Draw line between nearby bboxes iterate through redlist items
    for check in range(0, len(red_line_list)-1):					
        start_point = red_line_list[check] 
        end_point = red_line_list[check+1]
        check_line_x = abs(end_point[0] - start_point[0])   		# Calculate the line coordinates for x  
        check_line_y = abs(end_point[1] - start_point[1])			# Calculate the line coordinates for y
        # If both are We check that the lines are below our threshold distance.
        if (check_line_x < socialDistance) and (check_line_y < (socialDistance)):				
            cv2.line(bird_image, start_point, end_point, (255, 0, 0), 2)
            cv2.putText(bird_image, f'X: {start_point[0]} | Y: {start_point[1]}', (start_point[0], start_point[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,255,0), 1, cv2.LINE_AA) # Only above the threshold lines are displayed. 
            cv2.putText(bird_image, f'X: {end_point[0]} | Y: {end_point[1]}', (end_point[0], end_point[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,255,0), 1, cv2.LINE_AA) # Only above the threshold lines are displayed. 

    return warped_pts, bird_image, red_zone_list

def draw_perspective_boxes(img, warped_pts, dictionary, ids):

    #=============================================================================
    # Draws the bounding boxes for the perspective view, for a better user-sided
    # Experience
    #
    # Utilising the data from the birds-eye view within the red-zone list, we can
    # Draw RED bounding boxes around persons breaking social distance and
    # GREEN bounding boxes around persons adhering to it.
    # 
    # RETURN: Perspective image (frame)
    #
    #=============================================================================

    # Simple user-sided text to tell current amount of detections
    text = "People at Risk: %s" % str(len(ids)) 			
    location = (10,25)												
    cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246,86,86), 1, cv2.LINE_AA)

    detectionPercentage = (100/len(dictionary)) * len(ids)

    if detectionPercentage >= crowd_detection_threshold:
        color = (255, 0, 0)
        crowdText = "Detection Percentage: " + str(round(detectionPercentage)) + " [WARNING]"
        # With this warning, it may trigger an external alarm or warning
    else:
        color = (0, 255, 0)
        crowdText = "Detection Percentage: " + str(round(detectionPercentage)) 	

		
    location = (10,50)												
    cv2.putText(img, crowdText, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)  
    #=============================================================================

    # Similar to the birds eye view, draws boxes using the dictionary bounding box coords.
    for idx, pts in warped_pts.items():
        selected = dictionary[idx]
        if idx in ids:
            cv2.rectangle(img, (selected[2], selected[3]), (selected[4], selected[5]), (255, 0, 0), 2) # Create Red bounding selectedes  #starting point, ending point size of 2
        else:
            cv2.rectangle(img, (selected[2], selected[3]), (selected[4], selected[5]), (0, 255, 0), 2) # Create Green bounding boxes
    
    return img

# DarkNet and OpenCV related
netMain = None
metaMain = None
altNames = None
# Scaling for the birds-eye view
scale_w = 1.2 / 2
scale_h = 4 / 2
# used to record the time when we processed last frame

def SocialDistancing():

    #=============================================================================
    # ~~~~ YOLO OBJECT DETECTION RELATED ~~~~
    #=============================================================================
    
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

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

    #=============================================================================
    # Output the video's to these locations in this format 
    #=============================================================================

    perspective_out = cv2.VideoWriter(
            "./Output/perspective.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
            (new_width, new_height))

    birds_eye_view_out = cv2.VideoWriter(
            "./Output/birds_eye.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (int(new_width * scale_w), int(new_height * scale_h))
            )
    

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(new_width, new_height, 3)
    
    # Initialising Frame Numbers
    frame_num = 0 

    # Checks if it's the first frame to allow for marking points
    # (Though it is not the first frame but the third frame to allow some time for live footage)
    first_frame_display = True
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0

    while True:

        #=============================================================================
        # This section is for checking the footage's current frame rate.
        #=============================================================================
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        fps = int(fps)
        print("Current FPS: ", fps)
        #=============================================================================

        frame_num += 1
        ret, frame_read = cap.read()

        # Check if frame present :: 'ret' returns True if frame present, otherwise break the loop.
        # Pressing Q also quits the loop and saves the current feed
        if not ret or keyboard.is_pressed('q'):
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)

        if(len(mouse_pts) != 7):
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", get_mouse_points, param=frame_resized)
            

        if frame_num >= 3:
            if frame_num == 3 and first_frame_display:
                while True:
                    image = frame_resized
                    cv2.imshow("image", image)
                    cv2.waitKey(1)
                    if len(mouse_pts) == 7:
                        first_frame_display = False
                        cv2.destroyWindow("image")
                        break
                four_points = mouse_pts

                # Get perspective
                M, Minv = get_camera_perspective(frame_resized, four_points[0:4])

                socialDistancePts = [[four_points[4][0], four_points[4][1]], [four_points[5][0], four_points[5][1]]]
                socialDistance = get_socialdistance_birds_eye(socialDistancePts, M, scale_w, scale_h)
                
                print("Social Distance: ", socialDistance)
                
                pts = src = np.float32(np.array([four_points[4:]]))
                warped_pt = cv2.perspectiveTransform(pts, M)[0]
                d_thresh = np.sqrt(
                    (warped_pt[0][0] - warped_pt[1][0]) ** 2
                    + (warped_pt[0][1] - warped_pt[1][1]) ** 2
                )
                bird_image = np.zeros(
                    (int(frame_resized.shape[0] * scale_h), int(frame_resized.shape[1] * scale_w), 3), np.uint8
                )

            darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

            detections = darknet.detect_image(netMain, namesList, darknet_image, thresh=0.25)

            image, dictionary = cv_detect_persons(detections, frame_resized)

            warped_pts, bird_image, marked_ids = plot_points_on_bird_eye_view(
                frame_resized, dictionary, M, scale_w, scale_h, socialDistance
            )

            drawn_image = draw_perspective_boxes(image, warped_pts, dictionary, marked_ids)

            final_image = cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB)

            cv2.imshow('Perspective', final_image)
            cv2.imshow('Birds-eye View', bird_image)
            cv2.waitKey(3)
            
            perspective_out.write(final_image)
            birds_eye_view_out.write(bird_image)

    cap.release()
    perspective_out.release()
    birds_eye_view_out.release()
    print("::: Videos have been recorded under /Output/ :::")

if __name__ == "__main__":
    SocialDistancing()
