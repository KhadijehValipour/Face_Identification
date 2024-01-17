import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from create_face_bank import CreateFaceBank

class FaceIdentification :
    def __init__(self):
        self.model = None
        self.input_image = None
        self.results = None
        self.face_bank = None

    def load_model(self):
        self.model = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.model.prepare(ctx_id=0, det_size=(640, 640))
        
    
    def load_image(self, args):
        self.input_image = cv2.imread(args.image)
        self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)
        plt.imshow(self.input_image)
        
    
    def load_face_bank(self):
        self.results = self.model.get(self.input_image)
        self.face_bank = np.load("face_bank.npy", allow_pickle=True)
        
        
    def Identification(self):

        threshold= 25
        result_image = self.input_image.copy()
        for result in self.results:
            bbox = result.bbox

            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(result_image, (x, y), (w, h), (0, 255, 0), 2)

            for person in self.face_bank:
                face_bank_person_embedding = person["embedding"]
                new_person_embedding = result["embedding"]
                distance = np.sqrt(np.sum((face_bank_person_embedding - new_person_embedding)** 2))

                if distance < threshold:

                    cv2.putText(result_image, person["name"],
                                (int(bbox[0])-8 , int(bbox[1])-17),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (225, 0, 0), 2, cv2.LINE_AA)
                    break
            
            else:
                cv2.putText(result_image, "UnKnown",
                            (int(bbox[0])-8 , int(bbox[1])-30),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                
        plt.imshow(result_image)
        plt.show()



    def update_face_bank(self, args):
        CreateFaceBank.facebank(self.model, face_bank_path=args.update)
        self.face_bank = np.load("face_bank.npy", allow_pickle=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Face Identification')
    parser.add_argument('--image', type=str, required=True, help='path to input image')
    parser.add_argument('--update', action='store_true', help='update face bank with new embeddings')
    args= parser.parse_args()
 
    fi = FaceIdentification()
    fi.load_model()
    fi.load_image(args)


    if args.update:
        fi.update_face_bank(args)

    fi.load_face_bank()
    fi.Identification()