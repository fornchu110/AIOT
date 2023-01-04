import cv2
import numpy as np
import face_recognition
import pandas
import time
import threading
import win32gui
import win32con
# import matplotlib.pyplot as plt
# from IPython.display import display
# from PIL import Image

class ipcamCapture:
    def __init__(self, URL):
        self.Frame = []
        self.status = False
        self.isstop = False

	# 攝影機連接。
        self.capture = cv2.VideoCapture(URL)
    def start(self):
	    # 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print('ipcam started!\n')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
	# 記得要設計停止無限迴圈的開關。
        self.isstop = True
        print('ipcam stopped!\n')
   
    def getframe(self):
	# 當有需要影像時，再回傳最新的影像。
        return self.Frame.copy()
        
    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
        self.capture.release()

RED_COLOR = (255, 0, 0)
BLUE_COLOR = (0, 0, 255)
WHITE_COLOR = (255, 255, 255)
# 初始化known face list
known_face_list = list()

# 利用手機作為讀取影像的攝像頭, 裡面要放ip攝像頭帳號密碼和ip位置
# 手機熱點
# URL = "http://admin:admin@192.168.115.14:8081/" 
# 手機熱點2
URL = "http://admin:admin@192.168.155.89:8081/"
# 家裡wifi
# URL = "http://admin:admin@192.168.1.146:8081/" 

cap = ipcamCapture(URL)
cap.start()
time.sleep(1)

# 當攝像頭有開啟的時候
i = 0
print("known face start")
while(True):
    # 讀取一張攝像頭的圖片, cap.read()回傳兩個值, 是否讀到照片到ret, 讀到的照片到img
    img = cap.getframe()
    # 利用face_recognition辨識出人臉位置
    cur_face_locs = face_recognition.face_locations(img)
    # 利用位置畫出紅框
    if(cur_face_locs!=[]):
        # print("locs=", cur_face_locs)
        for cur_face_loc in cur_face_locs:
            # print("loc=", cur_face_loc)
            (y1, x2, y2, x1) = cur_face_loc
            # opencv用BGR, 所以設定藍色會出來紅色
            cv2.rectangle(img, (x1, y1), (x2, y2), BLUE_COLOR, 2)
    # cv2.imshow將看到的圖片顯示出來, imshow不是BGR不用轉換q
    cv2.imshow("Known face", img)
    # 將imshow置頂
    hwnd = win32gui.FindWindow(None, "Known face")#  
    CVRECT=cv2.getWindowImageRect("Known face")
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0,0,CVRECT[2],CVRECT[3], win32con.SWP_SHOWWINDOW)

    # cv2.waitKey用來等待是否有按鍵指令, 0xFF==ord()用來判斷是否接受到特定按鍵, ord()是轉換成ascii碼, 這邊以q為例  
    if cv2.waitKey(10) & 0xFF==ord('s'):
        userName = input("Name = ")
        i += 1
        # 配合cv2.imwrite可以做到照相功能, 將目前圖片存檔
        cv2.imwrite('known_user'+str(i)+'.jpeg', img)
        known_face_list.append(dict([('name', userName), ('filename', 'known_user'+str(i)+'.jpeg'), ('encode', None)]))
        # cv2.imshow("Save face ID", img)
    elif cv2.waitKey(10) & 0xFF==ord('q'):
        break

print("known list done\n")
cv2.destroyAllWindows()
# 讀取已知臉部資訊
for data in known_face_list:
    # imread用來讀圖片
    img = cv2.imread(data['filename'])
    # cvtColor用來轉換顏色ㄝ, imread是BGR所以轉成RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 透過face_recognition得到這個人像的特徵並儲存到encode, 限定只存一張臉一種特徵不然難以判斷
    data['encode'] = face_recognition.face_encodings(img)[0]

match_results_set = set()
# 獲取已知的人臉特徵
known_face_encodes = [data['encode'] for data in known_face_list]
# 容許的誤差, 誤差超過此值就無法辨識, 是unknown
tolerance = 0.5

# 開始對攝像頭收到的臉部做辨識
print("Face ID start")
while(True):
    match_results = []
    img = cap.getframe()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    cur_face_locs = face_recognition.face_locations(img)
    if(cur_face_locs!=[]):
        for cur_face_loc in cur_face_locs:
            (y1, x2, y2, x1) = cur_face_loc
            cv2.rectangle(img, (x1, y1), (x2, y2), BLUE_COLOR, 2)
    cv2.imshow("Face ID", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    cur_face_encodes = face_recognition.face_encodings(img, cur_face_locs)
    for cur_face_encode, cur_face_loc in zip(cur_face_encodes, cur_face_locs):
        face_distance_list = face_recognition.face_distance(known_face_encodes, cur_face_encode)
        # 取出測試圖片中臉部特徵的最小值, 再看與已知臉部特徵是否在誤差內
        min_distance_index = np.argmin(face_distance_list)
        # 與已知臉部在誤差內代表辨識出來
        if face_distance_list[min_distance_index]<tolerance:
            name = known_face_list[min_distance_index]['name']
            # 將第一次辨識到的畫面儲存下來
            if name not in match_results_set:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
                cv2.imwrite("match "+str(name)+'.jpeg', img)
            match_results_set.add(name)
        # 誤差外代表辨識不出來
        else:
            name = 'unknown'
    
        # 將辨識結果儲存下來
        match_results.append({
            'name': name,
            'location': cur_face_loc,
        })
        # 根據辨識結果和人臉位置畫出紅框和人名
        for match_result in match_results:
            y1, x2, y2, x1 = match_result['location']
            # 在臉部周圍畫出紅框
            cv2.rectangle(img, (x1, y1), (x2, y2), RED_COLOR, 2)
            # 畫一段紅色區域給辨識結果的文字做底
            # cv2.rectangle(img, (x1, y2 + 35), (x2, y2), RED_COLOR, cv2.FILLED)
            # 顯示辨識結果的文字
            cv2.putText(img, match_result['name'], (x1 + 10, y2 + 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, WHITE_COLOR, 2)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        cv2.imshow("Face ID", img)
        hwnd = win32gui.FindWindow(None, "Face ID")#  
        CVRECT=cv2.getWindowImageRect("Face ID")
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0,0,CVRECT[2],CVRECT[3], win32con.SWP_SHOWWINDOW)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.stop()
        break

cv2.destroyAllWindows()
# 將結果存成csv
df = pandas.DataFrame(list(match_results_set), columns = ["Student name"])
df.index += 1
print()
print("Result")
print(df)
df.to_csv("result.csv")
# read_csv = pandas.read_csv("result.csv")
# print(read_csv)