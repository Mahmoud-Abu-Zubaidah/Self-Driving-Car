import os # library to interact with the operating system
import cv2 # OpenCV library for image processing
import numpy as np # library for numerical operations
import polars as pl # DataFrame library
from shapely.geometry import Polygon, MultiPolygon # library for geometric objects


Base_lines = {
    "Lane1": {"line":[(450, 700), (0, 800)], "color":(0, 255, 0),'thickness':5},
    "Lane2": {"line":[(200, 870), (650, 720)], "color":(0, 0, 255),'thickness':5},
    "Lane3": {"line":[(1270, 900), (1000, 720)], "color":(0, 0, 255),'thickness':5},
    "Lane4": {"line":[(1350, 750), (1800, 900)], "color":(0, 255, 0),'thickness':5},
}
Areas = [
    [(200, 870), (650, 720), (1000, 720), (1270, 900)],
    [(1270, 900), (1000, 720), (1350, 750), (1800, 900)],
    [(200, 870), (0, 800), (450, 700), (650, 720)]
         ]

def draw_polygon(image, points, color, thickness=2):
    points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)

def lainDrawer(img, DrawLane=True,FrontArea=False):
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
        draw_polygon(line, lift_points, (100, 255, 0), 5)

        # Draw Right Shape
        right_points = [(1270, 900), (1000, 720), (1350, 750), (1800, 900)]
        draw_polygon(line, right_points, (100, 255, 0), 5)

        
        # Draw Middle Shape
        middle_points = [(200, 870), (650, 720), (1000, 720), (1270, 900)]
        draw_polygon(line, middle_points, (100, 50, 250), 5)
    

    return line


def intersections(image, rectangle_polygon,non_regular_polygon,color):
    intersection = rectangle_polygon.intersection(non_regular_polygon)
    # Draw the intersection area (if it exists)

    test = os.path.join('Tensorflow','workspace','images', 'video', 'frames' ,'ba6b36ba-d1c9-11ef-a928-601895437905.jpg')
    test = cv2.imread(test)
    
    if not intersection.is_empty:
        # If the intersection is a single polygon
        intersection_points = list(intersection.exterior.coords)
        draw_polygon(test, intersection_points, (0, 0, 255), thickness=5)  # Red for intersection
        cv2.imshow("Testing", cv2.resize(test, (800,600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#------
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[...,0] = color
        # 0 --> red
        # 60 --> green
        # 150 --> pink
        mask = np.zeros(image.shape[:2], dtype="uint8")
        Segmentation = image.copy()
        cv2.fillPoly(mask, np.array([intersection_points], dtype=np.int32), (255,255, 255))
        imask = mask > 0
        Segmentation[imask] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[imask]
        cv2.imshow("mask", cv2.resize(Segmentation, (800,600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("Segmentation.jpg", Segmentation)
#-------
        #Draw the intersection box area
        draw_polygon(image, intersection_points, (0, 0, 255), thickness=3)  # Red for intersection

def draw_intersection(top_objects, image, im_width, im_height):
    for box in top_objects:
        
        xmin, ymin, xmax, ymax = box
        (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                        int(ymin * im_height), int(ymax * im_height))
        rectangle_points = [(top, left),  (top,right), (bottom, right),(bottom, left)]
        rectangle_polygon = Polygon(rectangle_points)
        for area in Areas:
            if Areas.index(area) == 0:
                color = 0
            else:
                color = 60
            non_regular_polygon = Polygon(area)
            print(rectangle_polygon)
            # draw_polygon(image, rectangle_points, (255, 0, 0), thickness=5)  # Blue for rectangle
            # draw_polygon(image, area, (0, 255, 0), thickness=5)  # Green for non-regular shape
            intersections(image,rectangle_polygon,non_regular_polygon,color=color)

def Main(Image, top_objects, DrawLane,FrontArea):
    img = Image.copy()
    im_width, im_height = img.shape[0], img.shape[1]



    # Visualize base Lines
    img = lainDrawer(img, DrawLane,FrontArea)

    # Visualize Intersection
    draw_intersection(top_objects, img, im_width, im_height)
    return img

# Example usage Uncomment this to test the code.
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
#     img = Main(image, top_objects, DrawLane=True,FrontArea=False)
#     # # Save the image
#     cv2.imwrite("image2.jpg", img)