import cv2

# 加载YOLOv3模型
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
# 加载目标类别标签
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# 获取输出层信息
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# 加载视频
video_path = 'path_to_your_video.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 对帧进行目标检测
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 解析检测结果
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # 设置置信度阈值
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                # 计算边界框的左上角坐标
                x = int(center_x - width/2)
                y = int(center_y - height/2)

                # 保存检测结果
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])

    # 应用非最大抑制进行边界框过滤
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 绘制边界框和类别标签
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    if len(indices):
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = round(confidences[i] * 100, 2)
            color = colors[i]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 显示结果
    cv2.imshow('Object Tracking', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
# 3
