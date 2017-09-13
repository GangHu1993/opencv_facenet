人脸检测识别

人脸检测采用opencv自带检测模块，人脸识别采用facenet作为识别；

|--opencv_face_dect_recon
|----src 
|------collect_train_data.py	//采集数据
|------train_knn.py				//训练knn
|------camera_detect_recognition.py//基于摄像头的人脸检测和识别


src
--main()
---build sess->load face bable->dect face->face recon->predict.

Usage：
collecting train data:
1.打开train_dir文件夹,新建训练人文件夹，如“HuGang”。
2.打开collect_train_data.py,修改保存路径处，文件夹名字。
3. python collect_train_data.py
Note:采集图像时候，需要调整摄像头，使得采集的图像只有一个人物的正脸，尽量不要大幅度动摇，以影响采集图像质量。

tain knn:
1. python train_knn.py

cammear based face detect and recongination:
1. python camera_detect_recognition.py