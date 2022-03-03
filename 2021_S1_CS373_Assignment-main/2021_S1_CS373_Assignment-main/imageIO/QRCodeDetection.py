
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import numpy as np
import imageio


def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array



def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    

    greyscale_pixel_array = pixel_array_r

    for i in range(image_height):
        for j in range(image_width):
            greyscale_pixel_array[i][j] = round(pixel_array_r[i][j]*0.299+pixel_array_g[i][j]*0.587+pixel_array_b[i][j]*0.114)
    
    return greyscale_pixel_array



def computeMinAndMaxValues(pixel_array, image_width, image_height):

    minimum = pixel_array[0][0]

    maximum = minimum


    for row in pixel_array:

        if min(row) < minimum:

            minimum = min(row)

        if max(row) > maximum:

            maximum = max(row)

    return (minimum, maximum)
 

def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    min, max = computeMinAndMaxValues(pixel_array, image_width, image_height)
    if min == max:
        return greyscale_pixel_array
        
    k = round(255/(max-min), 3)
    for i in range(image_height):
        for j in range(image_width):
            num = pixel_array[i][j]
            if round(k*(num - min)) < 0:
                greyscale_pixel_array[i][j] = 0
            elif round(k*(num - min)) > 255:
                greyscale_pixel_array[i][j] = 255
            else:
                greyscale_pixel_array[i][j] = round(k*(num - min))
    
    return greyscale_pixel_array


def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    ouputArray = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            if (i==0 or i==image_height-1 or j==0 or j==image_width-1):
                ouputArray [i][j] = 0.000
            else:
                ouputArray [i][j] = abs((pixel_array[i-1][j+1]*(1)+pixel_array[i][j+1]*(2)+pixel_array[i+1][j+1]*(1)+ pixel_array[i-1][j-1]*(-1)+pixel_array[i][j-1]*(-2)+pixel_array[i+1][j-1]*(-1))/8.0)
    return ouputArray


def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    ouputArray = np.zeros((image_height,image_width))
   
    for i in range(image_height):
        for j in range(image_width):
            if (i==0 or i==image_height-1 or j==0 or j==image_width-1):
                ouputArray [i][j] = 0.000
            else:
                ouputArray [i][j] = abs((pixel_array[i-1][j-1]*(1)+pixel_array[i-1][j]*(2)+pixel_array[i-1][j+1]*(1)+pixel_array[i+1][j-1]*(-1)+pixel_array[i+1][j]*(-2)+pixel_array[i+1][j+1]*(-1))/8.0)
    return ouputArray


def computeBoxAveraging3x3(pixel_array, image_width, image_height):
    ouputArray = pixel_array
    for i in range(image_height):
        for j in range(image_width):
            if (i==0 or i==image_height-1 or j==0 or j==image_width-1):
                ouputArray [i][j] = 0.000
            else:
                ouputArray [i][j] = abs((pixel_array[i][j]+pixel_array[i-1][j]+pixel_array[i+1][j]+pixel_array[i-1][j-1]+pixel_array[i][j-1]+pixel_array[i+1][j-1]+pixel_array[i-1][j+1]+pixel_array[i][j+1]+pixel_array[i+1][j+1])/9.0)
    return ouputArray


def computeMedian5x3ZeroPadding(pixel_array, image_width, image_height):
    pixel_array1 = [[0 for i in range(image_width+4)]]
    for i in range(image_height):
        n = [0,0]+pixel_array[i]+[0,0]
        pixel_array1.append(n)
    pixel_array1.append([0 for i in range(image_width+4)])
    out=[[0 for x in range(image_width)] for y in range(image_height)]
    for i in range(1,image_height+1):
        for j in range(2,image_width+2):
            m = [pixel_array1[i-1][j-2],pixel_array1[i-1][j-1],pixel_array1[i-1][j],\
            pixel_array1[i-1][j+1],pixel_array1[i-1][j+2],pixel_array1[i][j-2],pixel_array1[i][j-1],\
            pixel_array1[i][j],pixel_array1[i][j+1],pixel_array1[i][j+2],pixel_array1[i+1][j-2],\
            pixel_array1[i+1][j-1],pixel_array1[i+1][j],pixel_array1[i+1][j+1],pixel_array1[i+1][j+2]]
            m.sort()
            out[i-1][j-2]=m[7]
    return out


def computeGaussianAveraging3x3RepeatBorder(pixel_array, image_width, image_height):
    list1 = [[0 for i in range(image_width)]for i in range(image_height)]
    for i in range(image_height):
        pixel_array[i].append(pixel_array[i][-1])
        pixel_array[i] =[pixel_array[i][0]] + pixel_array[i]
    pixel_array = [pixel_array[0]]+ pixel_array + [pixel_array[-1]]
    for i in range(image_height):
        for j in range(image_width):
            num1 = pixel_array[i][j] + 2*pixel_array[i][j+1] + pixel_array[i][j+2]
            num2 = 2*pixel_array[i+1][j] + 4*pixel_array[i+1][j+1] + 2*pixel_array[i+1][j+2]
            num3= pixel_array[i+2][j] + 2*pixel_array[i+2][j+1] + pixel_array[i+2][j+2]
            result = round((num1 + num2 +num3)/16,2)
            list1[i][j] = result
    return list1


def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] < threshold_value:
                pixel_array[i][j] = 0
            else:
                pixel_array[i][j] = 255
    return pixel_array


def BorderZeroPadding(pixel_array, image_width, image_height):
    numofline = image_height + 2
    lineelement = image_width + 2

    new_array = np.zeros((numofline,lineelement))
    i = 1
    
    while i < (numofline-1):
        j = 2
        while j < (lineelement - 1):
            new_array[i][j] = pixel_array[i-1][j - 1]
            j += 1
        i += 1
    return new_array


def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    output_array = pixel_array
    data_array = BorderZeroPadding(pixel_array, image_width, image_height)
    
    for i in range(image_height+2):
        for j in range(image_width):
            data_i = i + 1
            data_j = j + 1
            if (data_array[data_i - 1][data_j - 1] >= 1) and (data_array[data_i - 1][data_j] >= 1) and (data_array[data_i - 1][data_j + 1] >= 1) and (data_array[data_i][data_j - 1] >= 1) and(data_array[data_i][data_j] >= 1) and (data_array[data_i][data_j + 1] >= 1) and(data_array[data_i + 1][data_j - 1] >= 1) and (data_array[data_i + 1][data_j] >= 1) and(data_array[data_i + 1][data_j + 1] >= 1):
                output_array[i][j] = 1
    return output_array


def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):

    output_array = pixel_array
    data_array = BorderZeroPadding(pixel_array, image_width, image_height)
    
    for i in range(image_height):
        for j in range(image_width):
            data_i = i + 1
            data_j = j + 1
            if (data_array[data_i - 1][data_j - 1] >= 1) or (data_array[data_i - 1][data_j] >= 1) or (data_array[data_i - 1][data_j + 1] >= 1) or (data_array[data_i][data_j - 1] >= 1) or (data_array[data_i][data_j] >= 1) or (data_array[data_i][data_j + 1] >= 1) or(data_array[data_i + 1][data_j - 1] >= 1) or (data_array[data_i + 1][data_j] >= 1) or (data_array[data_i + 1][data_j + 1] >= 1):
                output_array[i][j] = 255
    return output_array




def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)

    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))


    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):

            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


def prepareRGBImageForImshowFromIndividualArrays(r,g,b,w,h):
    rgbImage = []
    for y in range(h):
        row = []
        for x in range(w):
            triple = []
            triple.append(r[y][x])
            triple.append(g[y][x])
            triple.append(b[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage
    


def writeGreyscalePixelArraytoPNG(output_filename, pixel_array, image_width, image_height):

    imageio.imwrite(r"dest.png",pixel_array)




def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageio.imread(input_filename)

    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))


    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):

            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


def sobel_suanzi(img):
    r, c = img.shape
    new_image = np.zeros((r, c))
    new_imageX = np.zeros(img.shape)
    new_imageY = np.zeros(img.shape)
    s_suanziX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])      
    s_suanziY = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])     
    for i in range(r-2):
        for j in range(c-2):
            new_imageX[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * s_suanziX))
            new_imageY[i+1, j+1] = abs(np.sum(img[i:i+3, j:j+3] * s_suanziY))
            new_image[i+1, j+1] = (new_imageX[i+1, j+1]*new_imageX[i+1,j+1] + new_imageY[i+1, j+1]*new_imageY[i+1,j+1])**0.5

    return np.uint8(new_image) 

tempLst=[]
size=0
lst=[]
def inLst(i,j):
    for temp in lst:
        if temp[0]==i and temp[1]==j:
            return True
    return False
def filltempLst(i,j,rows,cols,image,record):
    lst1=[[i,j]]
    myRecord=np.zeros(image.shape)
    while len(lst1)>0:
        temp_len=len(lst1)
        temp_i=lst1[temp_len-1][0]
        temp_j=lst1[temp_len-1][1]
        lst1.pop()

        if record[temp_i,temp_j]==1:
            tempLst.clear()
            return False
        tempLst.append([temp_i,temp_j])
        myRecord[temp_i,temp_j]=1

        get_i=temp_i-1
        get_j=temp_j
        if get_i>=0 and image[get_i,get_j]>0:
            if myRecord[get_i,get_j]==0:
                myRecord[get_i,get_j]=1
                lst1.append([get_i,get_j])
        get_i=temp_i+1
        get_j=temp_j
        if get_i<rows and image[get_i,get_j]>0:
            if myRecord[get_i,get_j]==0:
                myRecord[get_i,get_j]=1
                lst1.append([get_i,get_j])
        get_i=temp_i
        get_j=temp_j-1
        if get_j>=0 and image[get_i,get_j]>0:
            if myRecord[get_i,get_j]==0:
                myRecord[get_i,get_j]=1
                lst1.append([get_i,get_j])
        get_i=temp_i
        get_j=temp_j+1
        if get_j<cols and image[get_i,get_j]>0:
            if myRecord[get_i,get_j]==0:
                myRecord[get_i,get_j]=1
                lst1.append([get_i,get_j])
    for x in tempLst:
        record[x[0],x[1]]=1
    return True

def getMaxConnect(image,record):
    global size
    global lst
    [rows,cols]=image.shape
    for i in range(rows):
        for j in range(cols):
            if image[i,j]>0:
                if filltempLst(i,j,rows,cols,image,record):
                    if len(tempLst)>size:
                        size=len(tempLst)
                        lst=tempLst[:]
                tempLst.clear()


def main():
    filename = "poster1small.png"
    image = imageio.imread(filename)
    image_height,image_width,channels = image.shape
    px_array_r = image[:,:,0]
    px_array_g = image[:,:,1]
    px_array_b = image[:,:,2]
    

    gray_image = computeRGBToGreyscale(px_array_r,px_array_g,px_array_b,image_width,image_height)
    

    sobel = sobel_suanzi(gray_image)

    filter_image_1 = computeBoxAveraging3x3(sobel,image_width, image_height)

    filter_image_2 = computeBoxAveraging3x3(filter_image_1,image_width, image_height)

    threshold = computeThresholdGE(filter_image_2, 10, image_width, image_height)
    result = computeDilation8Nbh3x3FlatSE(threshold, image_width, image_height)

    record = np.zeros((image_height,image_width))
    getMaxConnect(result,record)
    
    arr=np.zeros(result.shape)
    for x in lst:
        arr[x[0],x[1]]=255
    
    min_w = image_width
    min_h = image_height
    max_w = 0
    max_h = 0
    for i in range(image_height):
        for j in range(image_width):
            if arr[i,j] == 255:
                if min_h > i:
                    min_h = i
                if min_w > j:
                    min_w = j
                if max_h < i:
                    max_h = i
                if max_w < j:
                    max_w = j
           
    pyplot.imshow(image)
    axes = pyplot.gca()
    print(max_w)
    print(max_h)
    rect = Rectangle((min_w, min_h), (max_w-min_w), (max_h-min_h), linewidth=3, edgecolor='g', facecolor='none' )

    axes.add_patch(rect)


    pyplot.show()
    


if __name__ == "__main__":
    main()