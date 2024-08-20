import cv2
import numpy as np
#AM: 58161
filename1 = '1.png'

img = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
print(img.shape)
imgrows, imgcols=img.shape

#Find the median value by the function Median_Value
def Median_Value(Kernel):
    frows = len(Kernel)
    fcols = len(Kernel[0])
    k = frows*fcols #size of kernel
    K = Kernel.reshape(1, k)
    K = np.sort(K)
    N = (k // 2)
    MedianValue = K[(0, N)]
    MedianRow = frows // 2
    MedianColumn = fcols // 2
    Kernel[MedianRow, MedianColumn] = MedianValue
    return(MedianValue)

#Create Kernel [3*3]
img2 = np.zeros(shape = (imgrows, imgcols))
Kernel = np.zeros((3, 3))
frows, fcols = Kernel.shape
MedianRow = frows // 2
MedianCol = fcols // 2
r = 0
c = 0
row = 0
col = 0

for r in range(imgrows - MedianRow - 1):
  for c in range(imgcols - MedianCol - 1):
   for row in range(frows):
    for col in range(fcols):

       Kernel[row, col] = img[r + row, c + col] #Find the new kernel
       col = col + 1

    row = row + 1
    col = 0

   MV = Median_Value(Kernel)               #Call the function to find the median Value of the new Kernel
   img2[r + MedianRow, c + MedianCol] = MV #Put the new pixel in the filtered image
   c = c + 1
   row = 0
  r = r + 1
  c = 0


img2_toshow = (img2).astype(np.uint8) #Convert the values from int to unit8
new_image = cv2.imwrite(r'new_image.png',img2_toshow) #Save the filtered image
filename = 'new_image.png'
new_image = cv2.imread(filename)
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
imgrows, imgcols = img.shape

#New image
img2 = np.zeros((imgrows, imgcols))
#Calulate the summed image
def calcimage(img):

    sumH = 0 #sum of the first row
    img2[0,0]=img[0,0]
    for col in range(0, imgcols):
        row = 0
        sumH =img[row, col] + img2[row, col-1]
        img2[row, col] = sumH
        col = col + 1

    for row in range(0, imgrows):
        col = 0
        sumV = img[row, col]+ img2[row-1,col]
        img2[row, col] = sumV
        row = row + 1
    r = 1
    sumX=0
    for r in range(1,imgrows):
        for c in range(0, imgcols):

         sumX = img2[r-1,c] + img2[r,c-1] - img2[r-1,c-1] + img[r,c]
         img2[r, c] = sumX
         c = c + 1
        r = r + 1
        c=0
    return(img2)

img2 = calcimage(img)

#Threshold of the binary image
ret, binary = cv2.threshold(img,15,255,  cv2.THRESH_BINARY)
cv2.imshow('Binary Threshold', binary)

#Function for integral of grey area
def CalcAreas(img2, x_A, y_A, x_D, y_D):

  x_B = x_A
  y_B = y_D
  x_C = x_D
  y_C = y_A

  Int_A = img2[y_A - 1,x_A - 1]
  Int_B = img2[y_B - 1,x_B - 1]
  Int_C = img2[y_C - 1,x_C - 1]
  Int_D = img2[y_D - 1,x_D - 1]

  Int_T = Int_A + Int_D - Int_B - Int_C

  return(Int_T)

# Apply the Component analysis function
analysis = cv2.connectedComponentsWithStats(binary,4,cv2.CV_32S)
(totalLabels, label_ids, values, centroid) = analysis


# Initialize a new image to store all the output components
output = np.zeros(img.shape, dtype="uint8")

# Loop through each component
area = np.zeros(totalLabels)

for i in range(1, totalLabels):
    area[i] = values[i, cv2.CC_STAT_AREA]
# Mean area to decide a threshold
mean_area = int(np.mean(area))
max_area = int(np.max(area))
j = 1
BoundingArea=np.zeros(totalLabels)
Mean_graylevel_value = np.zeros(totalLabels)
x_img2, y_img2 = img2.shape
for i in range(1, totalLabels):

  if (area[i] > 10) & (area[i] <= max_area):

    x1 = values[i, cv2.CC_STAT_LEFT]
    y1 = values[i, cv2.CC_STAT_TOP]
    w = values[i, cv2.CC_STAT_WIDTH]
    h = values[i, cv2.CC_STAT_HEIGHT]
    (X, Y) = centroid[i]

    # Coordinate of the bounding box
    pt1 = (x1,y1)
    pt2 = (x1+ w,y1+h)

    BoundingArea[j] = w*h
    bbox=int(BoundingArea[j])
    Mean_graylevel_value[j] = (CalcAreas(img2,x1,y1,w+x1,h+y1) / bbox)
    print("Region ",j)
    print("Area(px): ",int(area[i]))
    print("Bounding Box Area (px): ",int(BoundingArea[j]))
    print("Mean graylevel value in bounding box: ",Mean_graylevel_value[j])

    # Bounding boxes for each component
    output = cv2.rectangle(new_image, (x1,y1),(x1+w,y1+h),(0,0,255), 2)

    # Using cv2.putText()
    new_image = cv2.putText(img=new_image,text="{}".format(j),org=(int(X),int(Y)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 255),thickness = 2)
  i = i + 1
  j = j + 1

  #Function to colorize the cells with different color
  def imshow_components(labels):
      # Map component labels to hue val
      label_hue = np.uint8(179 * labels / np.max(labels))
      blank_ch = 255 * np.ones_like(label_hue)
      labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

      # cvt to BGR for display
      labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

      # set bg label to black
      labeled_img[label_hue == 0] = 0

      cv2.namedWindow('labeled', cv2.WINDOW_NORMAL)
      cv2.imshow('labeled', labeled_img)

imshow_components(label_ids)
cv2.imshow("Image", img)
cv2.imshow("Result", new_image)
#Save image
new_image2 = cv2.imwrite(r'RESULT.png',new_image)
cv2.waitKey(0)

