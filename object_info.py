import os # library to interact with the operating system
import cv2
import numpy as np
import math
from shapely.geometry import Polygon # library for geometric objects

Areas = [
    [(200, 870), (650, 720), (1000, 720), (1270, 900)],# Middle Area
    [(1270, 900), (1000, 720), (1350, 750), (1800, 900)],# Right Area
    [(200, 870), (0, 800), (450, 700), (650, 720)] # Left Area
         ]

def draw_polygon(image, points, color, thickness=2):
    """
    The function draws a polygon on the original image.
    NOTE: The function creates a copy from given image and returns the modified image.
    params:
    - image: the original image
    - points: list of points that define the polygon
    - color: color of the polygon
    - thickness: thickness of the polygon lines set as 2 by default
    """
    img = image.copy()
    points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [points], isClosed=True, color=color, thickness=thickness)
    return img

def lainDrawer(img, DrawLane=True,FrontArea=False):
    """
    The function draws the lanes on the image based on the given parameters, wheather to draw lines or areas or both.
    NOTE: The function creates a copy from given image and returns the modified image.
    params:
    - img: the original image
    - DrawLane: boolean to draw the lines or not
    - FrontArea: boolean to draw the areas or not
    """
    line = img.copy()

    # Draw lines. Numbers are From Left to Right
    if DrawLane:
        line = cv2.line(line, (200, 870), (650, 720), (0, 0, 255), 5)  # Line 2
        line = cv2.line(line, (1270, 900), (1000, 720), (0, 0, 255), 5)  # Line 3
        line = cv2.line(line, (450, 700), (0, 800), (0,255,0), 5)  # Line 1
        line = cv2.line(line, (1350, 750), (1800, 900), (0,255,0), 5)  # Line 4
    
    if FrontArea:
        # Draw Lift Shape
        lift_points = [(200, 870), (0, 800), (450, 700), (650, 720)]
        line = draw_polygon(line, lift_points, (100, 255, 0), 5)

        # Draw Right Shape
        right_points = [(1270, 900), (1000, 720), (1350, 750), (1800, 900)]
        line = draw_polygon(line, right_points, (100, 255, 0), 5)

        
        # Draw Middle Shape
        middle_points = [(200, 870), (650, 720), (1000, 720), (1270, 900)]
        line = draw_polygon(line, middle_points, (100, 50, 250), 5)
    
    return line



#Function to draw the intersection area boarders

def intersection_border(image, rectangle_polygon,non_regular_polygon,color):
    """
    The function draws the intersection area borders on the image.
    params:
    - image: the original image
    - rectangle_polygon: POLYGON object of the detected object
    - non_regular_polygon: POLYGON object of the area to extract the intersection from
    - color: color of the intersection area borders
    """
    img = image.copy()

    # B G R 
    color = (77, 77, 255) if not color else (0, 230, 255)

    intersection = rectangle_polygon.intersection(non_regular_polygon)
    # Draw the intersection area (if it exists)
    if not intersection.is_empty:
        # If the intersection is a single polygon
        try:
            intersection_points = list(intersection.exterior.coords)
        except:
            return img
        img = draw_polygon(img, intersection_points, color, thickness=3)  # Red for intersection
    return img


def draw_intersection(top_objects, image, im_width, im_height,draw_intersection_border,segment_object):
    """
    The function loops over the detected objects, then loops over all POLYGON areas and find which part of detected object intersects with each area.
    To do that, it loops over each detected object and denormalize the coordinates to the original image size then loop over each area and find the intersection border.
    
    param:
    - top_object: List of objects normalized coordinate
    - image: the original image
    - im_width: image width
    - im_height: image height
    - intersection_boarder: boolean to draw border or not
    """
    img = image.copy()
    for box in top_objects:
        xmin, ymin, xmax, ymax = box
        (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                        int(ymin * im_height), int(ymax * im_height))
        rectangle_points = [(top, left),  (top,right), (bottom, right),(bottom, left)]
        rectangle_polygon = Polygon(rectangle_points)

        for area in Areas:
            if Areas.index(area) == 0:
                color = 170
            else:
                color = 30
            non_regular_polygon = Polygon(area)
            if draw_intersection_border:
                img = intersection_border(img,rectangle_polygon,non_regular_polygon,color=color)
            if segment_object:
                img = segmentation(img,rectangle_polygon,non_regular_polygon,color)
    return img

def segmentation(image, rectangle_polygon,non_regular_polygon,color):
    """
    Segments the intersection area between two polygons (a rectangle and a non-regular polygon) 
    from the given image by applying a color mask to that region.

    Parameters:
    - image: The input image (numpy array) on which the segmentation will be performed.
    - rectangle_polygon: A Shapely polygon object representing the rectangular area.
    - non_regular_polygon: A Shapely polygon object representing the non-regular polygon.
    - color: The hue value (int) to which the intersection area will be changed in the HSV color space.

    Returns:
    - A new image (numpy array) with the intersection area of the polygons colored with the specified color.
    
    Notes:
    - If there is no intersection, the original image is returned without changes.
    - The color is applied using HSV color space manipulation.
    """

    img = image.copy()
    intersection = rectangle_polygon.intersection(non_regular_polygon)
    # Draw the intersection area (if it exists)
    if not intersection.is_empty:
        try:
            intersection_points = list(intersection.exterior.coords)
        except:
            return img

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[...,0] = color
        hsv[...,1] = hsv[...,1] * 0.7
        hsv[...,2] = 150 #I add it on 15/5/2025
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.fillPoly(mask, np.array([intersection_points], dtype=np.int32), (255,255, 255))
        imask = mask > 0
        img[imask] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[imask]
        # img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB) 
    return img 




def line_drawer(image, objects_coord):
    """
    The function draws a line from the center of the object to the center of the main object.
    param:
    - image: the original image
    - objects_coord: list of list coordinats [[xmin, ymin, xmax, ymax]] where coord are normalized
    Note:
        - The function return a copy of the original image
    
    """
    def calculate_distance(start_point, end_point):
        """
        Calculate the distance between two points.

        :param start_point: tuple (x1, y1)
        :param end_point: tuple (x2, y2)
        :return: distance between the points
        """
        x1, y1 = start_point
        x2, y2 = end_point
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance
    
    img = image.copy()
    im_width, im_height = img.shape[0], img.shape[1]
    center = (600, 950)

    for box in objects_coord:
        xmin, ymin, xmax, ymax = box
        (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                        int(ymin * im_height), int(ymax * im_height))
        
        w_distance = math.sqrt((bottom - top)**2)
        h_distance = math.sqrt((left - right)**2)

        center_x = int(top + w_distance/2)
        center_y = int(left + h_distance/2)

        #Define Car center 
        cv2.circle(img,center,5,(255,255,0),-1)

        cv2.circle(img,(center_x,center_y),5,(142,0,142),-1)

        cv2.line(img,center,(center_x,center_y),(142,0,142),3)

        # cv2.line(img,center,(center_x,center[1]),(142,0,142),3) # Center Line

        #The angle between the center of the object and the center of the image calculated but if you want to use it just return it
        Angle = math.atan2(center_y - center[1], center_x - center[0])
        Angle = round(np.abs(math.degrees(Angle)),2)

        if Angle > 90:
            Angle = 180 - Angle
        

    return img



def main(Image, top_objects, DrawLane, FrontArea,draw_intersection_border,segment_object,object_line):
    img = Image.copy()
    im_width, im_height = img.shape[0], img.shape[1]

    # Visualize base Lines
    
    img = draw_intersection(top_objects, img, im_width, im_height,draw_intersection_border,segment_object)
    img = lainDrawer(img, DrawLane,FrontArea)
    
    if object_line:
        img = line_drawer(img, top_objects)
    
    return img

# Test the code

# if __name__ == "__main__":
#     # Load the image
#     image = os.path.join('Tensorflow','workspace','images', 'video', 'frames' ,'ba6b36ba-d1c9-11ef-a928-601895437905.jpg')
#     image = cv2.imread(image)
#     # Load the top objects
#     top_objects = [[0.513058066368103, 0.5096254348754883, 0.8098193407058716, 0.7756397724151611],
#                    [0.5279399156570435, 0.45540735125541687, 0.613673210144043, 0.5090571045875549],
#                    [0.18736574053764343, 0.772793710231781, 0.7757474184036255, 1.0],
#                    [0.5343732833862305, 0.38254061341285706, 0.5905256271362305, 0.41385772824287415]]
#     # # Visualize the image
#     img = main(image, top_objects, DrawLane=0,FrontArea=0,draw_intersection_border=1,segment_object=1,object_line=1)

#     cv2.imshow("Result", cv2.resize(img, (800,600)))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()