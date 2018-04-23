from dlib import get_frontal_face_detector, shape_predictor


class FaceLandMarkDetection:
    def __init__(self):
        self._face_detector = get_frontal_face_detector()
        self._pose_estimator = shape_predictor(
            './models/shape_predictor_68_face_landmarks.dat')

    def find_face_locations(self, image, upsample_factor=1):
        """
        Return the locations of faces in the given image
            :param self:
            :param image: Image in which the face locations have to be determined
            :param upsample_factor=1: 
            :return: Returns a dlib iterable containing dlib rect objects of the face locations
        """
        return self._face_detector(image, upsample_factor)

    def find_face_landmarks(self, img, face_locations=None):
        """
        Returns the landmarks/parts for each face in img
            :param self: 
            :param img: Image for which the Face landmarks have to be returned
        """

        # The 1 in the second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        if face_locations is None:
            face_locations = self.find_face_locations(img)
        return [self._pose_estimator(img, d) for _, d in enumerate(face_locations)]
