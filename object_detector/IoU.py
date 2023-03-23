import pandas as pd
import csv
def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

ground_truth = pd.read_csv('/home/quocanhlee/Desktop/human-detector-master/object_detector/results/test_level5_yolov5.csv')
vali_data = ground_truth.iloc[:,1:6].values
print(vali_data)
print("=======================================================")

detection = pd.read_csv('/home/quocanhlee/Desktop/human-detector-master/object_detector/results/test_level5_SVM.csv')
detect_data = detection.iloc[:,:].values
print(detect_data)
print("=======================================================")


row1,col1 = vali_data.shape
row2, col2 = detect_data.shape

for i in range(row1):
    for j in range(row2):
        if (vali_data[i][4] == detect_data[j][4]):
            print(vali_data[i][4])
            boxA = [vali_data[i][0], vali_data[i][1], vali_data[i][2] +vali_data[i][0], vali_data[i][3] + vali_data[i][1]]
            boxB = [detect_data[j][0], detect_data[j][1], detect_data[j][2], detect_data[j][3]]
            IOU = bb_intersection_over_union(boxA, boxB)
            print(IOU)
            print("================================================================")
            data=[vali_data[i][4], IOU]
            with open(r'/home/quocanhlee/Desktop/human-detector-master/object_detector/IoU/IoU_level5.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(data)