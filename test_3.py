import cv2
import dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model
import time
import winsound

# 定义眼睛图像的尺寸
EYE_IMG_SIZE = (34, 26)

# 加载训练好的眨眼检测模型
blink_model = load_model('models/2024_04_16_04_32_02.keras')
blink_model.summary()

# 初始化人脸检测器和人脸关键点检测器
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 定义一个函数,用于从原始图像中裁剪出眼睛区域
def crop_eye(img, eye_points):
    # 找到眼睛区域的最小和最大坐标
    x_min, y_min = np.amin(eye_points, axis=0)
    x_max, y_max = np.amax(eye_points, axis=0)
    # 计算眼睛区域的中心坐标
    center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    # 计算眼睛区域的宽度和高度
    eye_width = (x_max - x_min) * 1.2
    eye_height = eye_width * EYE_IMG_SIZE[1] / EYE_IMG_SIZE[0]
    # 计算眼睛区域的边界坐标
    margin_x, margin_y = eye_width / 2, eye_height / 2
    min_x, min_y = int(center_x - margin_x), int(center_y - margin_y)
    max_x, max_y = int(center_x + margin_x), int(center_y + margin_y)
    # 将边界坐标转换为整数类型
    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)
    # 从原始图像中裁剪出眼睛区域
    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]
    return eye_img, eye_rect

cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('result', 1000, 750)

# 打开摄像头
cap = cv2.VideoCapture(0)

# 初始化眨眼次数,眼睛状态,起始时间,帧数计数器,帧率,疲劳检查时间,疲劳警告结束时间和眨眼历史
total_blinks = 0
eyes_closed = False  
start_time = time.time()
frame_count = 0
fps = 0
last_fatigue_check_time = time.time()  
fatigue_warning_end_time = 0
sleepy_warning_end_time = 0
blink_history = []

# 循环读取视频帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 缩小图像尺寸,加快处理速度
    frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)
    # 水平翻转图像
    frame = cv2.flip(frame, 1)
    # 复制原始图像,用于绘制结果
    result_frame = frame.copy()
    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 检测人脸
    faces = face_detector(gray)
    
    # 遍历每个检测到的人脸
    for face in faces:
        # 检测人脸关键点
        landmarks = landmark_predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        
        # 裁剪左眼和右眼区域
        le_img, le_rect = crop_eye(gray, eye_points=landmarks[36:42])
        re_img, re_rect = crop_eye(gray, eye_points=landmarks[42:48])
        
        # 提高眼睛图像的对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
        le_img = clahe.apply(le_img)
        re_img = clahe.apply(re_img)

        # 调整眼睛图像的尺寸
        le_img = cv2.resize(le_img, dsize=EYE_IMG_SIZE)
        re_img = cv2.resize(re_img, dsize=EYE_IMG_SIZE)
        
        # 水平翻转右眼图像
        re_img = cv2.flip(re_img, flipCode=1)
        
        # 将眼睛图像转换为模型输入格式
        le_input = le_img.reshape((1, EYE_IMG_SIZE[1], EYE_IMG_SIZE[0], 1))
        re_input = re_img.reshape((1, EYE_IMG_SIZE[1], EYE_IMG_SIZE[0], 1))
        
        # 使用模型预测左眼和右眼的睁闭状态
        le_pred = blink_model.predict(le_input)
        re_pred = blink_model.predict(re_input)
        
        le_status = 'O' if le_pred > 0.1 else '-'
        re_status = 'O' if re_pred > 0.1 else '-'
        
        le_prob = '%.2f' % le_pred
        re_prob = '%.2f' % re_pred
        
        cv2.rectangle(result_frame, pt1=tuple(le_rect[0:2]), pt2=tuple(le_rect[2:4]), color=(255,255,255), thickness=1)
        cv2.rectangle(result_frame, pt1=tuple(re_rect[0:2]), pt2=tuple(re_rect[2:4]), color=(255,255,255), thickness=1)
        
        # 计算左眼状态文本的位置
        le_status_pos = (le_rect[0], le_rect[1]-6)
        # 计算右眼状态文本的位置
        re_status_pos = (re_rect[0], re_rect[1]-6)
        
        # 计算左眼概率文本的位置
        le_prob_pos = (le_rect[0]-45, le_rect[3])
        # 计算右眼概率文本的位置
        re_prob_pos = (re_rect[2]+3, re_rect[3])
        
        cv2.putText(result_frame, le_status, le_status_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        cv2.putText(result_frame, re_status, re_status_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        
        cv2.putText(result_frame, le_prob, le_prob_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        cv2.putText(result_frame, le_prob, le_prob_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(result_frame, re_prob, re_prob_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        cv2.putText(result_frame, re_prob, re_prob_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        # 检测眨眼动作
        if le_status == '-' or re_status == '-':
            if not eyes_closed:
                eyes_closed = True
                blink_start_time = time.time()  # 记录闭眼开始时间
                total_blinks += 1
                blink_history.append(time.time())  # 将眨眼时刻添加到历史记录中
        else:
            eyes_closed = False
        
        # 计算眨眼次数文本的位置
        total_blinks_pos = (2, 20)
        if eyes_closed:
            cv2.putText(result_frame, f"Total Blinks: {total_blinks}", total_blinks_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            cv2.putText(result_frame, f"Total Blinks: {total_blinks}", total_blinks_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
        else:
            cv2.putText(result_frame, f"Total Blinks: {total_blinks}", total_blinks_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            cv2.putText(result_frame, f"Total Blinks: {total_blinks}", total_blinks_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        
        # 检查是否需要进行眨眼频率疲劳检测
        if time.time() - last_fatigue_check_time > 10:
            if len(blink_history) > 0:
                # 计算最早的眨眼时刻与当前时间的差值
                if time.time() - blink_history[-1] > 10:
                    # 如果超过10秒没有眨眼,显示疲劳提示信息
                    fatigue_warning_end_time = time.time() + 2  # 设置提示信息结束时间为2秒后
            else:
                # 如果过去10秒内没有眨眼记录,显示疲劳提示信息    
                fatigue_warning_end_time = time.time() + 2
            
            # 更新上一次检查时间 
            last_fatigue_check_time = time.time()
            
            # 移除10秒之前的眨眼记录
            while len(blink_history) > 0 and time.time() - blink_history[0] > 10:
                blink_history.pop(0)

        # 检测闭眼时间是否超过2秒,睡意检测
        if le_status == '-' and re_status == '-' and time.time() - blink_start_time > 2:
            # 播放警告音
            winsound.PlaySound("alert.wav", winsound.SND_ASYNC)
            # 更新眨眼开始时间
            blink_start_time = time.time()
            # 设置警告信息的结束时间为当前时间加上2秒
            sleepy_warning_end_time = time.time() + 2
        

    # 帧数计数器加1
    frame_count += 1
    
    # 计算当前时间与起始时间的差值
    elapsed_time = time.time() - start_time
    
    # 如果累计时间超过0.5秒,计算帧率,更新显示的帧率,并重置计数器和起始时间
    if elapsed_time > 0.5:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    
    # 在画面右上角显示检测频率
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(result_frame, fps_text, (result_frame.shape[1]-125, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(result_frame, fps_text, (result_frame.shape[1]-125, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
    
    # 如果当前时间在疲劳提示信息显示的时间范围内,显示提示信息
    if time.time() < fatigue_warning_end_time:  
        fatigue_warning_text_1 = "Blinking Frequency"
        fatigue_warning_text_2 = 'Too Low'
        (text1_width, text1_height), _ = cv2.getTextSize(fatigue_warning_text_1, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        fatigue_warning_pos_1 = (result_frame.shape[1] // 2 - text1_width // 2, result_frame.shape[0] // 2 - text1_height)
        (text2_width, text2_height), _ = cv2.getTextSize(fatigue_warning_text_2, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        fatigue_warning_pos_2 = (result_frame.shape[1] // 2 - text2_width // 2, result_frame.shape[0] // 2 + text2_height)
        cv2.putText(result_frame, fatigue_warning_text_1, fatigue_warning_pos_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(result_frame, fatigue_warning_text_2, fatigue_warning_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # 如果当前时间在睡意警告信息显示的时间范围内,显示警告信息
    if time.time() < sleepy_warning_end_time:
        sleepy_warning_text_1 = "Feeling sleepy?"
        sleepy_warning_text_2 = 'Wake up!'
        (text1_width, text1_height), _ = cv2.getTextSize(sleepy_warning_text_1, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        sleepy_warning_pos_1 = (result_frame.shape[1] // 2 - text1_width // 2, result_frame.shape[0] // 2 - text1_height)
        (text2_width, text2_height), _ = cv2.getTextSize(sleepy_warning_text_2, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        sleepy_warning_pos_2 = (result_frame.shape[1] // 2 - text2_width // 2, result_frame.shape[0] // 2 + text2_height)
        cv2.putText(result_frame, sleepy_warning_text_1, sleepy_warning_pos_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(result_frame, sleepy_warning_text_2, sleepy_warning_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # 显示处理后的图像
    cv2.imshow('result', result_frame)
    key = cv2.waitKey(1)
    
    # 如果按下'Q'键或者ESC键或者关闭窗口,则退出程序
    if key == ord('q') or key == 27 or cv2.getWindowProperty('result', cv2.WND_PROP_VISIBLE) < 1:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()