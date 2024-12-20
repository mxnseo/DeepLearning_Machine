import sys
import os
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QThread,pyqtSignal,Qt
from PyQt5.QtGui import QImage,QPixmap
from PIL import Image
import time
import classification_train
import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure



# UI 파일 연결
# 단, UI 파일은 Python 코드 파일과 같은 디렉토리에 위치해야 한다.
# 현재 파일의 디렉토리 경로
current_file_dir = os.path.dirname(__file__)

# UI 파일의 절대 경로 설정
ui_file_path = os.path.join(current_file_dir, 'new1.ui')

if not os.path.isfile(ui_file_path):
    print(f"UI file does not exist: {ui_file_path}")
else:
    # UI 파일 연결
    form_class = uic.loadUiType(ui_file_path)[0]


# Thread 클래스
class TrainThread(QThread):
    progress = pyqtSignal(str)  # Progress signal to update the GUI
    
    def __init__(self, args,window):
        super().__init__()
        self.args = args
        self.window=window
        self._running=True

    def run(self):
        original_stdout = sys.stdout  # 원래의 표준 출력(터미널)을 저장
        # 표준 출력을 OutputCapture로 리다이렉트하여 터미널 출력을 캡처
        sys.stdout = OutputCapture(self.progress) 
        # 학습 작업 시작
        classification_train.main(self.window, self.args)
        # 학습 작업이 끝나면 표준 출력을 원래대로 복구
        sys.stdout = original_stdout
class TrainThread_test(QThread):
    progress = pyqtSignal(str)  # Progress signal to update the GUI
    
    def __init__(self,window):
        super().__init__()
        self.window=window
        self._running=True

    def run(self):
        original_stdout = sys.stdout  # 원래의 표준 출력(터미널)을 저장
        # 표준 출력을 OutputCapture로 리다이렉트하여 터미널 출력을 캡처
        sys.stdout = OutputCapture(self.progress) 
        # 학습 작업 시작
        self.window.test()
        # 학습 작업이 끝나면 표준 출력을 원래대로 복구
        sys.stdout = original_stdout
        
    
# 터미널 출력을 캡처하고 pyqt 시그널로 전송하는 클래스
class OutputCapture:
    def __init__(self, signal):
        self.signal = signal

    def write(self, text):
        # 공백이 아닌 경우에만 시그널로 전송
        if text.strip():  
            self.signal.emit(text)

    def flush(self):
        pass
    
# 화면을 띄우는데 사용되는 Class 선언
class WindowClass(QTabWidget, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # QLabel에서 텍스트가 길어지면 자동으로 줄바꿈
        self.file_dirShow.setWordWrap(True)
        
        # dataset directory 버튼 클릭시 openFileDialog 메서드를 호출
        self.data_directory.clicked.connect(lambda:self.openFolderDialog('folder_path'))
        # output directory 버튼 클릭시 openFileDialog 메서드를 호출
        self.out_directory.clicked.connect(lambda:self.openFolderDialog('result_path'))
        # 모델학습 버튼 클릭시 run_command 메서드 호출
        self.model_teach.clicked.connect(self.run_command)
        self.model_teach_stop.clicked.connect(self.for_key)
        
        self.model_directory.clicked.connect(lambda:self.openFolderDialog("model_path"))
        self.test_directory.clicked.connect(lambda:self.openFolderDialog("test_folder_path"))
        self.training.clicked.connect(self.thread_open)
        
        # 훈련 그래프 도중 멈출 수 있게 하는 키
        self.key=False
            
        # 그래프
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        acc_widget = self.findChild(QWidget, 'graph_canvas')
        canvas_layout = QVBoxLayout(acc_widget)
        canvas_layout.addWidget(self.canvas)

        self.ax1 = self.figure.add_subplot(2, 1, 1)
        self.ax2 = self.figure.add_subplot(2, 1, 2)
        
        self.ax1.set_title('Loss')
        self.ax1.set_xlim(0, 50)  # x축 범위 설정
        self.ax1.set_ylim(0, 1)   # y축 범위 설정

        self.ax2.set_title('Accuracy')
        self.ax2.set_xlim(0, 50)  # x축 범위 설정
        self.ax2.set_ylim(0, 1)   # y축 범위 설정
        
        # 추론 그래프
        self.current_image_index = 0  # 현재 이미지 인덱스
        
        test_widget = self.findChild(QWidget, 'train_image')
        # QLabel 생성 (이미지를 표시할 위젯)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)  # QLabel 중앙 정렬
        
        self.result_label = self.findChild(QLabel, 'result_label')
        # 이전 이미지 버튼
        self.prev_img.clicked.connect(self.show_prev_image)
        # 다음 이미지 버튼
        self.next_img.clicked.connect(self.show_next_image)

        # 레이아웃에 QLabel과 버튼 추가
        layout = QVBoxLayout(test_widget)
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        
    def thread_open(self):
        self.thread = TrainThread_test(self)
        self.thread.progress.connect(self.display_test_output)  # TrainThread의 출력을 GUI에 연결
        self.thread.start()        
    def run_shell_command(self):
        self.classification_shell.clear()  # 텍스트 위젯을 비웁니다.
        self.worker.start()    # 스레드를 시작하여 명령어를 실행합니다.

    def display_output(self,text):
        self.classification_shell.append(text)  # 터미널 출력을 텍스트 위젯에 추가합니다.
    def display_test_output(self,text):
        self.classification_shell_2.append(text)  # 터미널 출력을 텍스트 위젯에 추가합니다.
        
    # 그래프 그리기
    def plot(self,x_arr,to_numpy_valid,to_numpy_train):
        self.figure.clear()
        
        # Create subplots
        self.ax1 = self.figure.add_subplot(2, 1, 1)
        self.ax1.plot(x_arr, to_numpy_train[0], '-', label='Train loss',marker='o')
        self.ax1.plot(x_arr, to_numpy_valid[0], '--', label='Valid loss',marker='o')
        handles, labels = self.ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax1.legend(by_label.values(), by_label.keys())
        
        self.ax2 = self.figure.add_subplot(2, 1, 2)
        self.ax2.plot(x_arr, to_numpy_train[1], '-', label='Train acc',marker='o')
        self.ax2.plot(x_arr, to_numpy_valid[1], '--', label='Valid acc',marker='o')
        handles, labels = self.ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax2.legend(by_label.values(), by_label.keys())
        
        self.ax1.set_title('Loss')
        self.ax2.set_title('Accuracy')
        # canvas 그리기
        self.canvas.draw()
    
    def for_key(self):
        self.key=True
    
    def stop_training(self):
        if hasattr(self, 'thread') and self.thread.isRunning() and self.key:
            return False
        return True
        
    def openFolderDialog(self,path_type):
        folder_path=QFileDialog.getExistingDirectory(self, "폴더 선택", "")
        if folder_path:
            if path_type == "folder_path":
                self.folder_path = folder_path
                self.file_dirShow.setText(f"Select Test File path: {self.folder_path}")
            elif path_type == "result_path":
                self.resultfolder_path = folder_path
                if(self.folder_path==""):
                    self.file_dirShow.setText(f"input: File not selected \noutput: {self.resultfolder_path}")  
                else:
                    self.file_dirShow.setText(f"input: {self.folder_path}\noutput: {self.resultfolder_path}")
            elif path_type == "model_path":
                self.model_path = folder_path
                self.file_dirShow2.setText(f"Select Test File path: {self.model_path}")
            elif path_type == "test_folder_path":
                self.test_folder_path = folder_path
                if(self.model_path==""):
                    self.file_dirShow2.setText(f"input: File not selected \noutput: {self.test_folder_path}")  
                else:
                    self.file_dirShow2.setText(f"input: {self.model_path}\noutput: {self.test_folder_path}")
    # def openFolderDialog_result(self):
            
    def openFolderDialog_result(self):
        # 파일 다이얼로그를 열어 사용자가 파일을 선택하도록 함
        self.resultfolder_path = QFileDialog.getExistingDirectory(self, "choice folder", "")
        self.file_dirShow.setText(f"input: {self.folder_path}\noutput: {self.resultfolder_path}")
    def run_command(self):
        
        # 실행할 명령어 정의
        self.key=False
        # 에포크
        self.epochs_spinBox = self.findChild(QSpinBox, 'epochs_spinBox') 
        epoch=self.epochs_spinBox.value()
        # -j
        self.worker_spinBox = self.findChild(QSpinBox, 'worker_spinBox') 
        worker=self.worker_spinBox.value()
        # learning rate
        self.lr_spinBox = self.findChild(QDoubleSpinBox, 'lr_spinBox') 
        lr=self.lr_spinBox.value()
        # model
        self.model_comboBox = self.findChild(QComboBox, 'model_comboBox') 
        model=self.model_comboBox.currentText()
        # weight
        self.weight_comboBox = self.findChild(QComboBox, 'weight_comboBox') 
        weight=self.weight_comboBox.currentText()
        # device
        self.device_comboBox = self.findChild(QComboBox, 'device_comboBox') 
        device=self.device_comboBox.currentText()
        
        if(self.resultfolder_path!="" and self.folder_path!=""):
            args = classification_train.get_args_parser(self.folder_path,epoch,worker,lr,model,weight,device,self.resultfolder_path)
            self.thread = TrainThread(args, self)
            self.thread.progress.connect(self.display_output)  # TrainThread의 출력을 GUI에 연결
            self.thread.start() 
            
    def test(self):
        directory_path = os.path.join(self.model_path, '..', 'train')
        # 클래스 레이블 이름 train 폴더 밑에 하위 디렉터리 이름을 찾아 지정
        class_label = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
        self.class_la=class_label
        # load pth model
        model = classification_train.torch.load(f"{self.model_path}\\model.pth", weights_only=False)
        # set model to inference mode
        model.eval()
        print(model)

        transform = classification_train.torchvision.transforms.Compose([      
                classification_train.torchvision.transforms.Resize(256),      # 이미지 크기 256x256으로 조정   
                classification_train.torchvision.transforms.CenterCrop(224),  # 중앙을 기준으로 224x224로 자르기 
                classification_train.torchvision.transforms.ToTensor(),       # 이미지를 텐서로 변환 
                classification_train.torchvision.transforms.Normalize(        # 정규화 
                mean=[0.485, 0.456, 0.406],        # 이미지 정규화 평균
                std=[0.229, 0.224, 0.225])         # 이미지 정규화 표준편차
        ])
        # 테스트 이미지 디렉토리 설정
        test_dir = self.test_folder_path
        classification_train.os.chdir(test_dir)   # 작업 디렉토리 변경
        self.list = classification_train.os.listdir(test_dir)
        files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
        file_num = len(files)
        acc_num = 0
        number=0
        self.y_pred_list=[]
        self.pred_list=[]
        for file in self.list:
            start_time = time.time()
            print(file)
            test_image = Image.open(file)
            img = transform(test_image)
            print(img.shape)
            img = img.to('cpu')
            with classification_train.torch.no_grad():
                pred = model(img.unsqueeze(0))
                print(pred)
                y_pred = classification_train.torch.argmax(pred)
                print(y_pred)
                print(class_label[y_pred])
            
            using_time = time.time() - start_time
            print(f"using_time : {using_time}")
            if class_label[y_pred] == class_label[0]:
                acc_num = acc_num + 1
            elif class_label[y_pred] == class_label[1]:
                acc_num = acc_num + 1
            number += 1
            test_image = test_image.resize((256, 256))  # 256x256으로 조정
            test_image = test_image.crop((16, 16, 240, 240))  # 중앙을 기준으로 224x224로 자르기
            
            self.y_pred_list.append(y_pred)
            self.pred_list.append(pred)
            # 처음 이미지를 QLabel에 표시
            if number>0:
                self.display_image(test_image)  # 처음 이미지 표시
            print(f"right_result : {acc_num}")
            print(f'acc : {acc_num / file_num}')
            self.current_image_index+=1
        self.current_image_index-=1

    def display_image(self, pil_image):
        # PIL 이미지를 크기 통일 (예: 224x224)
        target_size = (640, 640)  # 원하는 크기
        resized_image = pil_image.resize(target_size, Image.LANCZOS)  # LANCZOS 필터 사용
        # PIL 이미지를 NumPy 배열로 변환 (RGB 형식)
        rgb_image = classification_train.np.array(resized_image.convert('RGB'))  # RGB로 변환

        # NumPy 배열을 OpenCV 형식(BGR)으로 변환
        open_cv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # RGB에서 BGR로 변환

        # OpenCV 이미지를 QImage로 변환
        height, width,channel= open_cv_image.shape
        bytes_per_line = 3 * width
        qimage = QImage(open_cv_image.data, width, height, bytes_per_line, QImage.Format_BGR888)  # BGR로 설정

        # QImage를 QPixmap으로 변환
        pixmap = QPixmap.fromImage(qimage)

        # QLabel에 QPixmap 설정
        self.image_label.setPixmap(pixmap)
        #self.result_label.setText()
        file_name=f"file:{self.list[self.current_image_index]}\n"
        class_label=f"class_label: {self.class_la[self.y_pred_list[self.current_image_index]]}\n"
        y_pred=f"y_pred:{self.pred_list[self.current_image_index][0][self.y_pred_list[self.current_image_index]]:.3f}"
        self.result_label.setText(file_name + class_label + y_pred)
        
    def show_next_image(self):
        if self.current_image_index < len(self.list) - 1:
            self.current_image_index += 1
            test_image = Image.open(self.list[self.current_image_index])  # 이미지 열기
            self.display_image(test_image)

    def show_prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            test_image = Image.open(self.list[self.current_image_index])  # 이미지 열기
            self.display_image(test_image)
    
        
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # WindowClass의 인스턴스 생성
    myWindow=WindowClass()
    # 프로그램 화면을 보여주는 코드
    myWindow.show()
    # 프로그램을 이벤트 루프로 진입시키는(프로그램을 작동시키는) 코드
    sys.exit(app.exec_())