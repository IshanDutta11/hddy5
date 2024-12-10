pip install pyqt5 lxml --upgrade

git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
cd ..

git clone https://github.com/tzutalin/labelImg
cd labelImg
pyrcc5 -o libs/resources.py resources.qrc

cd yolov5
python train.py --img 640 --batch 16 --epochs 500 --data dataset.yaml --weights yolov5s.pt --workers 2
