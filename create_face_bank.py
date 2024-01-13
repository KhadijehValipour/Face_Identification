import os 
import cv2
import numpy as np
from insightface.app import FaceAnalysis


class CreateFaceBank():
    def __init__(self):
        pass

    def load_model(self):
        model = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        model.prepare(ctx_id=0, det_size=(640, 640))
        return model
    
    def facebank(self, model, face_bank_path):
        self.face_bank_path = face_bank_path
        face_bank = []
        for person_name in os.listdir(face_bank_path):
            file_path = os.path.join(face_bank_path, person_name)
            if os.path.isdir(file_path):
                for image_name in os.listdir(file_path):
                    if image_name != ".DS_Store":
                        image_path = os.path.join(file_path , image_name)
                        print(image_path)
                        image = cv2.imread(image_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        result = model.get(image)

                        if not result:
                            print("warning: no face detected in image")
                            continue
                        elif len(result) > 1:
                            print("warning: more than one face detected in image")
                            continue

                        embedding =result[0]['embedding']
                        my_dict ={"name": person_name , "embedding": embedding}
                        face_bank.append(my_dict)
        
        np.save("face_bank.npy", face_bank)