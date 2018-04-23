import cv2
import numpy as np

'''
We know that to warp a triangle to another triangle we will need to use the 
affine transform. In OpenCV, warpAffine allows you to apply an affine transform 
to an image, but not a triangular region inside the image. To overcome this 
limitation we find a bounding box around the source triangle and crop the 
rectangular region from the source image. We then apply an affine transform 
to the cropped image to obtain the output image. Finally, we create a triangular 
mask by filling pixels inside the output triangle with white. This mask when
multiplied with the output image turns all pixels outside the triangle black 
while preserving the color of all pixels inside the triangle.
'''


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, tri1, tri2):

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(tri1)
    r2 = cv2.boundingRect(tri2)

    # Offset points by left top corner of the respective rectangles
    tri1Cropped = []
    tri2Cropped = []

    for i in xrange(0, 3):
        tri1Cropped.append(((tri1[0][i][0] - r1[0]), (tri1[0][i][1] - r1[1])))
        tri2Cropped.append(((tri2[0][i][0] - r2[0]), (tri2[0][i][1] - r2[1])))

    # Crop input image
    img1Cropped = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(
        np.float32(tri1Cropped), np.float32(tri2Cropped))

    # Apply the Affine Transform just found to the src image
    img2Cropped = cv2.warpAffine(img1Cropped, warpMat, (
        r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)

    img2Cropped = img2Cropped * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]
        :r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped


