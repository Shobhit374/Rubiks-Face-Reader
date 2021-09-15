# IMPORTING NECESSARY LIBRARIES

import numpy as np
import cv2
import os

# DEFINING LOWER AND UPPER BOUNDS FOR COLORS OF RUBIKS CUBE

red_lower = np.array([0, 87, 111], np.uint8)
red_upper = np.array([10, 255, 255], np.uint8)
green_lower = np.array([32, 52, 72], np.uint8)
green_upper = np.array([80, 255, 255], np.uint8)
blue_lower = np.array([100, 80, 2], np.uint8)
blue_upper = np.array([130, 255, 255], np.uint8)
white_lower = np.array([0, 0, 200], np.uint8)
white_upper = np.array([180, 25, 255], np.uint8)
yellow_lower = np.array([20, 100, 100], np.uint8)
yellow_upper = np.array([30, 255, 255], np.uint8)
orange_lower = np.array([10, 100, 100], np.uint8)
orange_upper = np.array([16, 255, 255], np.uint8)

# USING OS LIBRARY TO READ INPUT IMAGES FROM FOLDER

for root, dirs, files in os.walk("input/"):
    for name in files:
        path = os.path.join(root, name)
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (1024, 768))
        hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # CREATING MASKS FOR DIFFERENT COLORS

        red_mask = cv2.inRange(hsvimg, red_lower, red_upper)
        green_mask = cv2.inRange(hsvimg, green_lower, green_upper)
        blue_mask = cv2.inRange(hsvimg, blue_lower, blue_upper)
        white_mask = cv2.inRange(hsvimg, white_lower, white_upper)
        yellow_mask = cv2.inRange(hsvimg, yellow_lower, yellow_upper)
        orange_mask = cv2.inRange(hsvimg, orange_lower, orange_upper)
        kernal = np.ones((5, 4), "uint8")

        # PERFORMING MORPHOLOGICAL OPERATIONS FOR BETTER CONTOURS

        green_mask = cv2.dilate(green_mask, kernal, iterations=1)
        blue_mask = cv2.dilate(blue_mask, kernal, iterations=1)
        white_mask = cv2.dilate(white_mask, kernal, iterations=1)
        yellow_mask = cv2.dilate(yellow_mask, kernal, iterations=2)
        orange_mask = cv2.dilate(orange_mask, kernal, iterations=1)
        red_mask = cv2.dilate(red_mask, kernal, iterations=1)
        blank = np.zeros([768, 1024, 3], dtype=np.uint8)
        blank[:, :] = [0, 0, 0]  # creating a blank image
        contours, hierarchy = cv2.findContours(
            white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1 * peri, True)
            corners = len(approx)
            if(3000 < area < 50000 and corners == 4):
                x, y, w, h = cv2.boundingRect(c)
                img = cv2.rectangle(
                    blank, (x, y), (x + w-40, y + h-40), (255, 255, 255), -1)  # finding and drawing rectangles on white colour
        contours, hierarchy = cv2.findContours(
            red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1 * peri, True)
            corners = len(approx)
            if(3000 < area < 50000 and corners == 4):
                x, y, w, h = cv2.boundingRect(c)
                img = cv2.rectangle(
                    blank, (x, y), (x + w-40, y + h-40), (0, 0, 255), -1)  # finding and drawing rectangles on red colour

        contours, hierarchy = cv2.findContours(
            yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1 * peri, True)
            corners = len(approx)
            if(3000 < area < 50000 and corners == 4):
                x, y, w, h = cv2.boundingRect(c)
                img = cv2.rectangle(
                    blank, (x, y), (x + w-40, y + h-40), (0, 255, 255), -1)  # finding and drawing rectangles on yellow colour
        contours, hierarchy = cv2.findContours(
            green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1 * peri, True)
            corners = len(approx)
            if(3000 < area < 50000 and corners == 4):
                x, y, w, h = cv2.boundingRect(c)
                img = cv2.rectangle(
                    blank, (x, y), (x + w-40, y + h-40), (0, 255, 0), -1)  # finding and drawing rectangles on green colour
        contours, hierarchy = cv2.findContours(
            blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1 * peri, True)
            corners = len(approx)
            if(3000 < area < 50000 and corners == 4):
                x, y, w, h = cv2.boundingRect(c)
                img = cv2.rectangle(
                    blank, (x, y), (x + w-40, y + h-40), (255, 0, 0), -1)  # finding and drawing rectangles on blue colour
        contours, hierarchy = cv2.findContours(
            orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1 * peri, True)
            corners = len(approx)
            if(3000 < area < 50000 and corners == 4):
                x, y, w, h = cv2.boundingRect(c)
                img = cv2.rectangle(
                    blank, (x, y), (x + w-40, y + h-40), (0, 165, 255), -1)  # finding and drawing rectangles on orange colour
        gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(
            gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # finding contours of the drawn rectangles

        # FINDING CENTROID COORDINATES OF THE RECTANGLES

        vertical = []
        horizontal = []
        coords = []
        for cnt in contours:
            mom = cv2.moments(cnt)
            y = int(mom['m01']/mom['m00'])
            x = int(mom['m10']/mom['m00'])
            coords.append((x, y))
            vertical.append(y)
            horizontal.append(x)
        vertical = sorted(vertical, reverse=True)
        horizontal = sorted(horizontal)
        bot_row = vertical[:3]
        mid_row = vertical[3:6]
        top_row = vertical[6:9]
        col1 = horizontal[:3]
        col2 = horizontal[3:6]
        col3 = horizontal[6:9]
        order = []

        # FINDING COORDINATES OF CENTROID FOR EACH TILE

        for i in top_row:
            for j in col1:
                for it, (x, y) in enumerate(coords):
                    if(j, i) == (x, y):
                        order.append((j, i))
                        coords[it] = (0, 0)
        for i in top_row:
            for j in col2:
                for it, (x, y) in enumerate(coords):
                    if(j, i) == (x, y):
                        order.append((j, i))
                        coords[it] = (0, 0)
        for i in top_row:
            for j in col3:
                for it, (x, y) in enumerate(coords):
                    if(j, i) == (x, y):
                        order.append((j, i))
                        coords[it] = (0, 0)
        for i in mid_row:
            for j in col1:
                for it, (x, y) in enumerate(coords):
                    if(j, i) == (x, y):
                        order.append((j, i))
                        coords[it] = (0, 0)
        for i in mid_row:
            for j in col2:
                for it, (x, y) in enumerate(coords):
                    if(j, i) == (x, y):
                        order.append((j, i))
                        coords[it] = (0, 0)
        for i in mid_row:
            for j in col3:
                for it, (x, y) in enumerate(coords):
                    if(j, i) == (x, y):
                        order.append((j, i))
                        coords[it] = (0, 0)
        for i in bot_row:
            for j in col1:
                for it, (x, y) in enumerate(coords):
                    if(j, i) == (x, y):
                        order.append((j, i))
                        coords[it] = (0, 0)
        for i in bot_row:
            for j in col2:
                for it, (x, y) in enumerate(coords):
                    if(j, i) == (x, y):
                        order.append((j, i))
                        coords[it] = (0, 0)
        for i in bot_row:
            for j in col3:
                for it, (x, y) in enumerate(coords):
                    if(j, i) == (x, y):
                        order.append((j, i))
                        coords[it] = (0, 0)

        # FINDING COLOUR OF EACH TILE

        colors = []
        for k in order:
            j = k[0]
            i = k[1]
            colors.append((blank[i, j, 0], blank[i, j, 1], blank[i, j, 2]))
        output = []
        for x in colors:
            if x not in output:
                output.append(x)
        # ARRANGING COLORS AS PER OUTPUT FORMAT
        temp = {i: j for j, i in enumerate(output, start=1)}
        res = [temp[i] for i in colors]
        L = [str(res[0]), str(res[1]), str(res[2]), "\n", str(res[3]), str(res[4]),
             str(res[5]), "\n", str(res[6]), str(res[7]), str(res[8]), "\n"]
        # WRITING THE OUTPUT FILE
        file = open("Output/output_"+str(path[6:-4]+"txt"), "w")
        file.writelines(L)
        file.close
