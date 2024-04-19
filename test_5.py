import cv2
import dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model
import time
import winsound

# 定义眼睛图像的尺寸
EYE_IMG_SIZE = (34, 26)

# 加载预训练的眨眼检测模型
blink_model = load_model('models/2024_04_16_04_32_02.keras')
blink_model.summary()

# 初始化人脸检测器和面部特征点检测器
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_tracker = dlib.correlation_tracker()


# 从原始图像中裁剪出眼睛区域
def crop_eye(img, eye_points):
    # 计算眼睛区域的边界框
    x_min, y_min = np.amin(eye_points, axis=0)
    x_max, y_max = np.amax(eye_points, axis=0)
    # 计算眼睛区域的中心坐标
    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
    # 根据眼睛宽高比计算裁剪区域的宽度和高度
    eye_width = (x_max - x_min) * 1.2
    eye_height = eye_width * EYE_IMG_SIZE[1] // EYE_IMG_SIZE[0]
    # 计算裁剪区域的边界坐标
    margin_x, margin_y = eye_width // 2, eye_height // 2
    min_x, min_y = int(center_x - margin_x), int(center_y - margin_y)
    max_x, max_y = int(center_x + margin_x), int(center_y + margin_y)
    # 对边界框坐标进行数值修正,确保其为整数
    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)
    # 从灰度图中裁剪出眼睛区域
    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]
    return eye_img, eye_rect

# 对裁剪出的眼睛图像进行预处理
def preprocess_eye(eye_img, is_right_eye=False):
    # 使用自适应直方图均衡化提高图像对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    eye_img = clahe.apply(eye_img)
    # 缩放图像到指定尺寸
    eye_img = cv2.resize(eye_img, dsize=EYE_IMG_SIZE)
    # 如果是右眼图像,需要水平翻转
    if is_right_eye:
        eye_img = cv2.flip(eye_img, flipCode=1)
    return eye_img

# 检测眨眼并更新相关状态变量
def detect_blink(le_img, ri_img):
    global total_blinks, eyes_closed, blink_start_time, blink_history
    # 将眼睛图像转换为模型输入格式,使用模型预测左眼和右眼的睁闭状态
    le_pred = blink_model.predict(le_img.reshape((1, EYE_IMG_SIZE[1], EYE_IMG_SIZE[0], 1)))[0, 0]
    ri_pred = blink_model.predict(ri_img.reshape((1, EYE_IMG_SIZE[1], EYE_IMG_SIZE[0], 1)))[0, 0]
    # 根据预测结果判断眼睛状态
    le_status = 'O' if le_pred > 0.1 else '-'
    ri_status = 'O' if ri_pred > 0.1 else '-'
    if le_status == '-' or ri_status == '-':
        if not eyes_closed:
            eyes_closed = True
            blink_start_time = time.time()  # 记录闭眼开始时间
            total_blinks += 1
            blink_history.append(time.time())  # 将眨眼时刻添加到眨眼历史记录中
    else:
        eyes_closed = False
    return le_pred, le_status, ri_pred, ri_status

# 在图像上绘制眼睛状态和概率信息
def draw_eye_status(result_frame, le_rect, re_rect, le_status, re_status, le_pred, re_pred):
    # 绘制眼睛区域的边界框
    cv2.rectangle(result_frame, pt1=tuple(le_rect[0:2]), pt2=tuple(le_rect[2:4]), color=(255,255,255), thickness=1)
    cv2.rectangle(result_frame, pt1=tuple(re_rect[0:2]), pt2=tuple(re_rect[2:4]), color=(255,255,255), thickness=1)
    
    # 计算左右眼状态文本的位置
    le_status_pos = (le_rect[0], le_rect[1]-6)
    re_status_pos = (re_rect[0], re_rect[1]-6)
    
    # 计算左右眼睁闭概率文本的位置
    le_prob_pos = (le_rect[0]-45, le_rect[3])
    re_prob_pos = (re_rect[2]+3, re_rect[3])
    
    # 在图像上绘制左右眼状态和睁闭概率的文本
    cv2.putText(result_frame, le_status, le_status_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
    cv2.putText(result_frame, re_status, re_status_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
    
    cv2.putText(result_frame, f'{le_pred:.2f}', le_prob_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(result_frame, f'{le_pred:.2f}', le_prob_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(result_frame, f'{re_pred:.2f}', re_prob_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(result_frame, f'{re_pred:.2f}', re_prob_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

# 检测长时间闭眼(睡意检测)
def check_sleepy(le_status, re_status):
    global blink_start_time, last_face_time, sleepy_warning_end_time
    if le_status == '-' and re_status == '-' and time.time() - blink_start_time > 2 and time.time() - last_face_time > 2:
        winsound.PlaySound("alert.wav", winsound.SND_ASYNC)  # 播放警示音
        blink_start_time = time.time()  # 更新闭眼开始时间
        sleepy_warning_end_time = time.time() + 2  # 设置睡意警告信息的结束时间

# 检测低眨眼频率(疲劳检测)  
def check_blink_freq():
    global last_freq_check_time, freq_warning_end_time, last_face_time
    if time.time() - last_freq_check_time > 10:
        if (len(blink_history) == 0 or time.time() - blink_history[-1] > 10) and time.time() - last_face_time > 10:
        # 如果过去10秒内没有眨眼记录或者最近一次眨眼时刻与当前时间差超过10秒,且最后一次检测到人脸的时间与当前时间差超过10秒
            freq_warning_end_time = time.time() + 2  # 设置疲劳提示信息的结束时间
        last_freq_check_time = time.time()  # 更新上一次检查时间
        # 移除眨眼历史记录中10秒前的数据
        while len(blink_history) > 0 and time.time() - blink_history[0] > 10:
            blink_history.pop(0)

# 在图像上绘制睡意警告信息
def draw_sleepy_warning(result_frame):
    if time.time() < sleepy_warning_end_time:
    # 如果当前时间在睡意警告信息显示的时间范围内,显示警告信息
        sleepy_warning_text_1 = "Feeling sleepy?"
        sleepy_warning_text_2 = 'Wake up!'
        (text1_width, text1_height), _ = cv2.getTextSize(sleepy_warning_text_1, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        sleepy_warning_pos_1 = (result_frame.shape[1] // 2 - text1_width // 2, result_frame.shape[0] // 2 - text1_height - 30)
        (text2_width, text2_height), _ = cv2.getTextSize(sleepy_warning_text_2, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        sleepy_warning_pos_2 = (result_frame.shape[1] // 2 - text2_width // 2, result_frame.shape[0] // 2 + text2_height - 40)
        cv2.putText(result_frame, sleepy_warning_text_1, sleepy_warning_pos_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(result_frame, sleepy_warning_text_2, sleepy_warning_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

# 在图像上绘制疲劳提示信息
def draw_freq_warning(result_frame):
    if time.time() < freq_warning_end_time:  
    # 如果当前时间在疲劳提示信息显示的时间范围内,显示提示信息
        freq_warning_text_1 = "Blinking Frequency"
        freq_warning_text_2 = 'Too Low'
        (text1_width, text1_height), _ = cv2.getTextSize(freq_warning_text_1, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        freq_warning_pos_1 = (result_frame.shape[1] // 2 - text1_width // 2, result_frame.shape[0] // 2 - text1_height + 50)
        (text2_width, text2_height), _ = cv2.getTextSize(freq_warning_text_2, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        freq_warning_pos_2 = (result_frame.shape[1] // 2 - text2_width // 2, result_frame.shape[0] // 2 + text2_height + 40)
        cv2.putText(result_frame, freq_warning_text_1, freq_warning_pos_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(result_frame, freq_warning_text_2, freq_warning_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

# 在图像上绘制总眨眼次数
def draw_total_blinks(result_frame):
    total_blinks_pos = (2, 20)
    if eyes_closed:
        cv2.putText(result_frame, f"Total Blinks: {total_blinks}", total_blinks_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(result_frame, f"Total Blinks: {total_blinks}", total_blinks_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
    else:
        cv2.putText(result_frame, f"Total Blinks: {total_blinks}", total_blinks_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(result_frame, f"Total Blinks: {total_blinks}", total_blinks_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

# 计算并绘制实时帧率
def calc_and_draw_fps(result_frame):
    global frame_count, elapsed_time, fps, start_time
    
    frame_count += 1  # 帧数计数器加1
    elapsed_time = time.time() - start_time  # 计算当前帧与第一帧的时间差
    
    if elapsed_time > 0.5:
        fps = frame_count / elapsed_time  # 计算实时帧率
        frame_count = 0  # 重置帧数计数器
        start_time = time.time()  # 重置起始时间
    
    # 在图像上绘制实时帧率
    cv2.putText(result_frame, f"FPS: {fps:.2f}", (result_frame.shape[1]-125, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(result_frame, f"FPS: {fps:.2f}", (result_frame.shape[1]-125, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

# 创建一个名为'result'的可调整大小的窗口,用于显示处理后的视频帧
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('result', 1000, 750)

# 打开默认的摄像头
cap = cv2.VideoCapture(0)

# 初始化疲劳检测、帧率相关的状态变量
total_blinks = 0
eyes_closed = False  
start_time = time.time()
frame_count = 0
fps = 0
last_freq_check_time = time.time()  
freq_warning_end_time = 0
sleepy_warning_end_time = 0
blink_history = []
face_detected = False
last_face_time = 0
tracking_face = False  # 添加人脸跟踪状态标志
face_box = None  # 存储人脸边框的变量


# 视频处理循环
while cap.isOpened():
    ret, frame = cap.read()  # 读取一帧图像
    if not ret:
        break  # 如果读取失败,跳出循环

    # 预处理
    frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5) # 缩小图像尺寸,加快处理速度
    frame = cv2.flip(frame, 1)  # 水平翻转图像
    result_frame = frame.copy() # 复制原始图像,用于绘制结果
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图

    if not tracking_face:
        # 如果没有跟踪人脸,则进行人脸检测
        faces = face_detector(gray)
        if len(faces) > 0:
            if not face_detected:
                face_detected = True
                last_face_time = time.time()  # 记录最后一次检测到人脸的时间
            
            # 选择检测到的第一张人脸进行跟踪
            face = faces[0]
            face_tracker.start_track(frame, dlib.rectangle(face.left(), face.top(), face.right(), face.bottom()))
            tracking_face = True
        else:
            face_detected = False
            face_box = None  # 如果没有检测到人脸,将face_box设为None
    else:        
        # 如果正在跟踪人脸,则更新跟踪器
        tracking_quality = face_tracker.update(frame)
        
        if tracking_quality >= 7:
            # 如果跟踪质量足够高,则继续跟踪
            tracked_position = face_tracker.get_position()
            t_left = int(tracked_position.left())
            t_top = int(tracked_position.top())
            t_right = int(tracked_position.right())
            t_bottom = int(tracked_position.bottom())
            face_box = dlib.rectangle(t_left, t_top, t_right, t_bottom)
        else:
            # 如果跟踪质量太低,则停止跟踪,下一帧重新检测人脸
            tracking_face = False
            face_box = None  # 如果跟踪失败,将face_box设为None
            continue
        
    if face_box is not None:
        # 如果face_box不为None,即检测到人脸或跟踪成功
        if tracking_face:
            # 如果正在跟踪,绘制绿色边框
            cv2.rectangle(result_frame, (face_box.left(), face_box.top()), (face_box.right(), face_box.bottom()), (0, 255, 0), 2)
        else:
            # 如果没有在跟踪,绘制白色边框
            cv2.rectangle(result_frame, (face_box.left(), face_box.top()), (face_box.right(), face_box.bottom()), (255, 255, 255), 2)

        # 对跟踪到的人脸进行处理
        landmarks = landmark_predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        # 对检测到的每一张人脸进行处理
        
        # 裁剪左眼和右眼区域
        le_img, le_rect = crop_eye(gray, eye_points=landmarks[36:42])
        ri_img, ri_rect = crop_eye(gray, eye_points=landmarks[42:48])
        
        # 对裁剪出的眼睛图像进行预处理
        le_img = preprocess_eye(le_img)
        ri_img = preprocess_eye(ri_img, is_right_eye=True)

        # 检测眨眼并更新相关状态
        le_pred, le_status, ri_pred, ri_status = detect_blink(le_img, ri_img)

        # 在图像上绘制眼睛状态和睁闭概率
        draw_eye_status(result_frame, le_rect, ri_rect, le_status, ri_status, le_pred, ri_pred)
        
        # 检测长时间闭眼和低眨眼频率
        check_sleepy(le_status, ri_status)
        check_blink_freq()

    # 在图像上绘制睡意和疲劳警告信息
    draw_sleepy_warning(result_frame)
    draw_freq_warning(result_frame)

    # 在图像上绘制总眨眼次数
    draw_total_blinks(result_frame)

    # 计算并绘制实时帧率
    calc_and_draw_fps(result_frame)

    # 显示处理后的视频帧
    cv2.imshow('result', result_frame)
    key = cv2.waitKey(1)
    
    # 如果按下'q'键或者'Esc'键,或者关闭显示窗口,则退出程序
    if key == ord('q') or key == 27 or cv2.getWindowProperty('result', cv2.WND_PROP_VISIBLE) < 1:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()