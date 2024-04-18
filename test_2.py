import cv2
import dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model
import time
import winsound

# 定义眼睛图像的尺寸
IMG_SIZE = (34, 26)

# 加载训练好的眨眼检测模型
model = load_model('models/2024_04_16_04_32_02.keras')
model.summary()

# 初始化人脸检测器和人脸关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 定义一个函数,用于从原始图像中裁剪出眼睛区域
def crop_eye(img, eye_points):
    # 找到眼睛区域的最小和最大坐标
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    # 计算眼睛区域的中心坐标
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    # 计算眼睛区域的宽度和高度
    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]
    # 计算眼睛区域的边界坐标
    margin_x, margin_y = w / 2, h / 2
    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)
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
blink_count = 0
eyes_closed = False  
start_time = time.time()
frame_count = 0
fps = 0
last_check_time = time.time()  
warning_end_time = 0
warning_end_time_2s = 0
blink_history = []

# 循环读取视频帧
while cap.isOpened():
    ret, img_ori = cap.read()
    if not ret:
        break
    
    # 缩小图像尺寸,加快处理速度
    img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)
    # 水平翻转图像
    img_ori = cv2.flip(img_ori, 1)
    # 复制原始图像,用于绘制结果
    img = img_ori.copy()
    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 检测人脸
    faces = detector(gray)
    
    # 遍历每个检测到的人脸
    for face in faces:
        # 检测人脸关键点
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)
        
        # 裁剪左眼和右眼区域
        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])
        
        # 提高眼睛图像的对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
        eye_img_l = clahe.apply(eye_img_l)
        eye_img_r = clahe.apply(eye_img_r)

        # 调整眼睛图像的尺寸
        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        
        # 水平翻转右眼图像
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)
        
        # 将眼睛图像转换为模型输入格式
        eye_input_l = eye_img_l.reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1))
        eye_input_r = eye_img_r.reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1))
        
        # 使用模型预测左眼和右眼的睁闭状态
        pred_l = model.predict(eye_input_l)
        pred_r = model.predict(eye_input_r)
        
        state_l = 'O' if pred_l > 0.1 else '-'
        state_r = 'O' if pred_r > 0.1 else '-'
        
        prob_l = '%.2f' % pred_l
        prob_r = '%.2f' % pred_r
        
        cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=1)
        cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=1)
        
        # 计算左眼状态文本的位置
        state_pos_l = (eye_rect_l[0], eye_rect_l[1]-6)
        # 计算右眼状态文本的位置
        state_pos_r = (eye_rect_r[0], eye_rect_r[1]-6)
        
        # 计算左眼概率文本的位置
        prob_pos_l = (eye_rect_l[0]-40, eye_rect_l[3])
        # 计算右眼概率文本的位置
        prob_pos_r = (eye_rect_r[2]+5, eye_rect_r[3])
        
        cv2.putText(img, state_l, state_pos_l, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        cv2.putText(img, state_r, state_pos_r, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        
        cv2.putText(img, prob_l, prob_pos_l, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(img, prob_r, prob_pos_r, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # 检测眨眼动作
        if state_l == '-' or state_r == '-':
            if not eyes_closed:
                eyes_closed = True
                blink_start_time = time.time()  # 记录闭眼开始时间
                blink_count += 1
                blink_history.append(time.time())  # 将眨眼时刻添加到历史记录中
        else:
            eyes_closed = False
        
        # 检查是否需要进行眨眼频率疲劳检测
        if time.time() - last_check_time > 10:
            if len(blink_history) > 0:
                # 计算最早的眨眼时刻与当前时间的差值
                if time.time() - blink_history[0] > 10:
                    # 如果超过10秒没有眨眼,显示疲劳提示信息
                    warning_end_time = time.time() + 3  # 设置提示信息结束时间为3秒后
            else:
                # 如果过去10秒内没有眨眼记录,显示疲劳提示信息    
                warning_end_time = time.time() + 3
            
            # 更新上一次检查时间 
            last_check_time = time.time()
            
            # 移除10秒之前的眨眼记录
            while len(blink_history) > 0 and time.time() - blink_history[0] > 10:
                blink_history.pop(0)

        # 检测闭眼时间是否超过2秒
        if state_l == '-' and state_r == '-' and time.time() - blink_start_time > 2:
            # 播放警告音
            winsound.PlaySound("alert.wav", winsound.SND_ASYNC)
            # 更新眨眼开始时间
            blink_start_time = time.time()
            # 设置警告信息的结束时间为当前时间加上3秒
            warning_end_time_2s = time.time() + 3
            
        # 计算眨眼次数文本的位置
        blink_pos = (0, 20)
        if eyes_closed:
            cv2.putText(img, f"Blink Count: {blink_count}", blink_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            cv2.putText(img, f"Blink Count: {blink_count}", blink_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
        else:
            cv2.putText(img, f"Blink Count: {blink_count}", blink_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            cv2.putText(img, f"Blink Count: {blink_count}", blink_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        

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
    cv2.putText(img, fps_text, (img.shape[1]-125, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(img, fps_text, (img.shape[1]-125, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
    
    # 如果当前时间在疲劳提示信息显示的时间范围内,显示提示信息
    if time.time() < warning_end_time:  
        text = "Fatigue Warning: Blinking Frequency Too Low"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        warning_pos = (img.shape[1] // 2 - text_width // 2, img.shape[0] // 2 - text_height // 2)
        cv2.putText(img, text, warning_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    # 如果当前时间在警告信息的显示时间内,显示警告信息
    if time.time() < warning_end_time_2s:
        # 显示警告文字
        text = "Feeling sleepy? Wake up!"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        warning_pos = (img.shape[1] // 2 - text_width // 2, img.shape[0] // 2 - text_height // 2)
        cv2.putText(img, text, warning_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
    # 显示处理后的图像
    cv2.imshow('result', img)
    key = cv2.waitKey(1)
    
    # 如果按下'q'键或者关闭窗口,退出程序
    if key == ord('q') or cv2.getWindowProperty('result', cv2.WND_PROP_VISIBLE) < 1:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()