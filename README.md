# Indian Actor Recognition using Facenet and SVM
![Home Page](https://github.com/Nagasai524/indianActorRecognition/blob/main/readme_images/1.png)
## About the Project

This project aims to recognize Indian actors using the powerful FaceNet model for face recognition and Support Vector Machines (SVM) for classification. It's an exciting application of deep learning in the world of entertainment and facial recognition. 

You can quickly test the project on [Streamlit Cloud]()

The project is also available as a [kaggle kernel](https://www.kaggle.com/nagasai524/indian-actor-recognition-using-facenet). 

## Entire Flow of the project
![Flow Chart](https://github.com/Nagasai524/indianActorRecognition/blob/main/readme_images/flow_chart.png)

### Built With

- Python
- TensorFlow/Keras
- Scikit-learn
- OpenCV
- [FaceNet](https://github.com/davidsandberg/facenet)

### Data
I have collected a few datasets available on the web that have images of Indian actors. The collected images are then cropped such that the image contains only the face of the actor. For detecting the face in the image, I have used MTCNN, which is a state-of-the-art technique for detecting faces in an image. If you want to explore the process of extracting the cropped faces from images, You can refer to one of my [Kaggle kernel](https://www.kaggle.com/code/nagasai524/cropping-faces-from-indian-actor-images-with-mtcnn) that has entire process explained in it.
The cropped images of actor's faces used for this project are uploaded to Kaggle datasets and can be found here. [Link](https://www.kaggle.com/datasets/nagasai524/indian-actor-faces-for-face-recognition)

### Images of Deployed Version
#### Page containing the sample images
![Flow Chart](https://github.com/Nagasai524/indianActorRecognition/blob/main/readme_images/2.png)
#### Page with a search bar for searching the trained actors
![Flow Chart](https://github.com/Nagasai524/indianActorRecognition/blob/main/readme_images/3.png)
#### Output of the model for an actor not trained by the model.
![Flow Chart](https://github.com/Nagasai524/indianActorRecognition/blob/main/readme_images/4.png)
#### Output of the model for an actor trained by the model.
![Flow Chart](https://github.com/Nagasai524/indianActorRecognition/blob/main/readme_images/5.png)
![Flow Chart](https://github.com/Nagasai524/indianActorRecognition/blob/main/readme_images/6.png)

