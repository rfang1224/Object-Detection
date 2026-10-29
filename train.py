from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo26n.pt')
    results = model.train(
        data='HomeObjects-3K.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        name='household_model'
    )
