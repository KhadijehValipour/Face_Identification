# Face Identification

![Alt text](assents/1_qaymSPbmvevOhDAokVLbEw.png)

This technique is commonly used in various applications such as facial recognition systems, facial feature recognition, and emotion recognition. This helps to separate the face from the rest of the image, allowing for more accurate and efficient processing. By drawing frames around faces, it becomes easier to track and analyze facial features, expressions, and other relevant information. This technique plays an important role in various computer vision and image processing applications.


## How to create face bank

We will create a face bank folder containing people whose names are spoken by the device. The reason for this naming is that the neural network and mlp are not supposed to be trained with it, we only extract the features of each person's face.

```
jupyter nbconvert --to python create_face_bank.ipynb
```


## How to install

```
pip install -r requirements.txt
```

## How to use

```
jupyter nbconvert --to python face_identification.ipynb
```

### results

![Alt text](output/output1.png)


![Alt text](output/output2.png)

![Alt text](output/output3.png)