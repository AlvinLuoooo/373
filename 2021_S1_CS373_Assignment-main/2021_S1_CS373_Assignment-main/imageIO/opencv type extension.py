
import cv2



def obtain_gray_image(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    return gray


def obtain_sobel_image(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)  
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)  

    absX = cv2.convertScaleAbs(x)  
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  
    return dst



def obtain_mean_filter_image(img):
    img_blur = cv2.blur(img, (3, 3))
    img_blur = cv2.blur(img_blur, (4, 4))
    return img_blur


def obtain_threshold_image(img):
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    return binary


def obtain_contour_image(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x = 0
    y = 0
    w = 0
    h = 0
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
    return x,y,w,h



def main():
    filename = "poster1small.png"
    src = cv2.imread(filename)

    img = obtain_gray_image(src)

    dst = obtain_sobel_image(img)

    blur = obtain_mean_filter_image(dst)

    binary = obtain_threshold_image(blur)

    x,y,w,h = obtain_contour_image(binary)
    if w != 0:
        cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 5)
    cv2.imshow("result image", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()