from face_recon import Face_recon
import cv2

img=cv2.imread('t.JPG',cv2.IMREAD_COLOR)

face_recon = Face_recon()
res = face_recon.det_face_reco(img)
res1 = face_recon.det_face_reco(img)
#print (res)
