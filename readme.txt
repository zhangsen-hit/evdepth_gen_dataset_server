从rosbag中生成FastDepth-EV数据集的全部步骤

准备步骤：rosbag录制 /livox/lidar  /livox/imu  /dvs/events  /rendring(可选)等话题
	运行FastLIO2 获得 odometry.txt(rostopic echo /Odometry > odometry.txt) 与 scans.pcd *放入a文件夹*
	运行python events_npz.py得到data npz *放入b文件夹*
	从bag 文件得到video文件（python fetch_video.py 注意fps）    *放入d文件夹*

步骤a：     根据odometry.txt 与 scans.pcd 为每一个位姿生成一个 深度图 depth.npz
	python batch_generate_depth.py
	
步骤b：     根据odometry.txt中的时间戳，聚合事件信息，为每一个时刻生成一个 事件帧 events_tensor.npz
	python events_npz.py 
	python stack.py 
	
步骤c：	根据一一对应的深度图与事件帧，构建数据集
	python build_depth_dataset.py

步骤d：	可视化与检查数据集
	python fetch_depth.py  (52行调整所选深度图)
	python fetch_evframe.py  （18、19行调整所选ev frame）
	python fetch_frame.py    (44行调整所选frame)
	python show.py





