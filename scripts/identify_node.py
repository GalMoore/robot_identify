#!/usr/bin/env python
from __future__ import print_function

# import roslib
# roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import face_recognition
import numpy as np
from os.path import expanduser
home = expanduser("~") + "/"

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("identified_people_video",Image,queue_size=10)
    # will publish either: 1. no face found = empty string 2. "unknown face found" 3. name of person 
    self.who_can_see_now = rospy.Publisher('/identified_people_string_name', String, queue_size=10)
    self.bridge = CvBridge()
    # subscribes to input video from video_stream_opencv that is running
    self.image_sub = rospy.Subscriber("/camera/image_raw",Image,self.callback,queue_size=10)
    self.process_this_frame = True

    ## BUILD UP DATABASE OF FACE EMBEDDINGS AND NAMES
    # Load a sample picture 
    Gal_path = home + "catkin_ws/src/robot_identify/faces/gal.jpg"
    Gal_image = face_recognition.load_image_file(Gal_path)
    Gal_face_encoding = face_recognition.face_encodings(Gal_image)[0]
    # Load a second sample picture 
    biden_path = home + "catkin_ws/src/robot_identify/faces/biden.jpg"
    biden_image = face_recognition.load_image_file(biden_path)
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

    # Create arrays of known face encodings and their names
    self.known_face_encodings = [
        Gal_face_encoding,
        biden_face_encoding
    ]
    self.known_face_names = [
        "Gal Moore",
        "Joe Biden"
    ]
    ##################################################################################

    self.face_locations = []
    self.face_encodings = []
    self.face_names = []

  def callback(self,data):

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    frame = cv_image
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) # Resize frame of video to 1/4 size
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if self.process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        # if no faces found publish that there is no name
        if(len(self.face_locations)==0):
          self.who_can_see_now.publish("")

        self.face_names = []
        for face_encoding in self.face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown face detected"

            # # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            self.face_names.append(name)

    self.process_this_frame = not self.process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # if known person found publish name to topic
        self.who_can_see_now.publish(name)

    # window for debug
    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('robot_identify', anonymous=False)

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)