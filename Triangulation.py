import cv2


class Delaunay:
    def __init__(self, points, rect):
        self._points = points
        self._rect = rect

    def get_traingles(self):
        subdiv = cv2.Subdiv2D(self._rect)
        for p in self._points:
            subdiv.insert(p)

        return subdiv.getTriangleList()
