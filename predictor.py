import streamlit as st
st.set_page_config(page_title="Actor Detector",layout="wide",initial_sidebar_state='expanded')
from mtcnn.mtcnn import MTCNN
import numpy as np
import cv2
import joblib
import facenet
from sklearn.metrics.pairwise import cosine_similarity

faces_in_each_row = 4

@st.cache_resource()
def return_utils():
    """
    Function that return all required artifacts for the project.
    """
    model_emb = facenet.InceptionResNetV1(input_shape=(None, None, 3),classes=128)
    model_emb.load_weights('utils/facenet_keras_weights.h5')
    svm_model=joblib.load('utils/svm_classifier.sav')
    label_encoder=joblib.load('utils/label_encoder.sav')
    classes=label_encoder.classes_
    data=np.load('utils/data.npz')
    known_face_embeddings=data['a']
    detector=MTCNN()
    return model_emb,svm_model,label_encoder,detector,known_face_embeddings,classes

@st.cache_data(persist="disk")
def get_pixels(img):
  """
  Function reads the given image and identifies all the faces in the image using MTCNN. 
  MTCNN is the state of the art technique for face detection in images.
  Returns the list of coordinates for detected faces in the original image as well as list of cropped images for detected faces.
  """
  img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  results=detector.detect_faces(img)
  if not(results):
      return [],[]
  face_coords=[]
  faces=[]
  for i in range(len(results)):
    x1,y1,width,height=results[i]['box']
    x1,y1=abs(x1),abs(y1)
    x2,y2=x1+width,y1+height
    face=img[y1:y2,x1:x2]
    face=cv2.resize(face,(160,160))
    face_coords.append([x1,x2,y1,y2])
    faces.append(face)
  return face_coords,faces

@st.cache_data(persist="disk")
def get_embeddings(face_pixels):
    """
    Functions that generatest the embeddings for a given face using Facenet.
    Facenet identifies 128 features for a given face which can be considered as an embedding for a given face.
    """
    face_pixels=face_pixels.astype('float32')
    mean,std=face_pixels.mean(),face_pixels.std()
    face_pixels=(face_pixels-mean)/std
    samples=np.expand_dims(face_pixels,axis=0)
    yhat=model_emb.predict(samples)
    return yhat[0]

st.sidebar.markdown("### Navigation")
model_emb,svm_model,label_encoder,detector,known_face_embeddings,classes=return_utils()
choice=st.sidebar.radio("page_navigator",['Actor Recognizer','Test Images','Trained Actors'],label_visibility='collapsed')

st.sidebar.markdown("<h1 style='text-align:center'><u><b> Developer </b></u></h1>",unsafe_allow_html=True)
with st.sidebar:
    st.image("test_images/me.jpg")
    with st.expander("About Me"):
        st.write("""
            Hello, I'm a Data Scientist with 2 years of experience in data analysis, machine learning, and deep learning. I have a strong background in Computer Science which allows me to apply a wide range of techniques to solve complex business problems.

            I have a passion for understanding and solving complex data problems, and I pride myself on my ability to communicate technical concepts to non-technical stakeholders. I believe that the key to success in data science is to approach problems with curiosity, rigor, and creativity, and to continuously learn and adapt to new technologies and techniques.
        """)
    st.subheader("Social Links")
    col1,col2,col3=st.columns(3)
    col1.markdown("<a href='https://www.linkedin.com/in/nagasai-biginepalli-64648a146/'>Linkedin</a>",unsafe_allow_html=True)
    col2.markdown("<a href='https://github.com/Nagasai524'>Github</a>",unsafe_allow_html=True)
    col3.markdown("<a href='mailto:www.biginepallinagasai109@gmail.com'>Gmail</a>",unsafe_allow_html=True)
    
if choice=='Actor Recognizer':
  st.markdown("# <center> <u> Indian Actor Recognition </u> </center> <br/> <br/>",True)
  st.markdown("## <center> Upload an Image File </center>",True)
  a=st.file_uploader("file_uploader",label_visibility='collapsed',type=['jpg','jpeg','png','webp','JPG','JPEG','PNG','WEBP'],accept_multiple_files=False)
  cnt1,cnt2=st.columns(2)
  if a:
    with cnt1:
      st.markdown("### <center>Uploaded Image</center>",True)
      st.image(a)
    file_bytes = np.asarray(bytearray(a.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    face_coords,face_pixels=get_pixels(image)
    if len(face_coords)==0:
        st.error("No Face detected in Image.")
    else:
        with st.spinner('Extracting Embeddings for detected faces'):
            actor_names=[]
            filtered_face_pixels=[]
            for i in range(len(face_coords)):
                x1,x2,y1,y2=face_coords[i][0],face_coords[i][1],face_coords[i][2],face_coords[i][3]
                cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),4)
                face_embedding=get_embeddings(face_pixels[i])
                actor_index=svm_model.predict([face_embedding])

                # Getting Probability Values for each class

                # probs_svc = svm_model.decision_function([face_embeddings])
                # probs_svc = (probs_svc - probs_svc.min()) / (probs_svc.max() - probs_svc.min())
                # print(f"Max Probability: {probs_svc[0][actor_index[0]]}")
            
                distance=cosine_similarity([face_embedding],[known_face_embeddings[actor_index[0]]])
                distance=distance[0]
                #print(distance)
                actor_name=label_encoder.inverse_transform(actor_index)
                actor_name=actor_name[0]
                if distance>0.48:
                    actor_names.append(actor_name)
                    cv2.putText(image,actor_name,(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,255),2)
                    filtered_face_pixels.append(face_pixels[i])

        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #image=cv2.resize(image,(700,700))
        with cnt2:
          st.markdown("### <center>Processed Image</center>",True)
          st.image(image)
        if len(actor_names)==1:
            st.markdown("### <center><u> Detected Actor </u></center>",True)
            st.success("The Recognized actor in the image is "+actor_name)
        elif len(actor_names)>1:
            st.markdown("### <center><u> Detected Actors </u></center>",True)
            detected_actors=",".join(actor_names)
            st.success("The Recognized actors in the image are "+detected_actors)
        elif len(actor_names)==0:
            st.error("No trained actor is recorgnised in the provided image. Go to 'Trained Actors' page to check if the actor was listed there.")
        
        for i in range(len(filtered_face_pixels)):
            if i%faces_in_each_row==0:
                container=st.container()
                col1,col2,col3,col4=container.columns(faces_in_each_row)
                column_array=[col1,col2,col3,col4]
            with container:
                with column_array[i%faces_in_each_row]:
                    st.image(filtered_face_pixels[i])
                    st.write(actor_names[i])
        st.balloons()

elif choice=='Test Images':
  st.markdown("### <center>Sample Test Images</center>",True)
  with st.container():
    cnt1,cnt2,cnt3,cnt4=st.columns(4)
    with cnt1:
        st.image('test_images/chiranjeevi.jpeg')
        st.write('Chiranjeevi')
    with cnt2:
        st.image('test_images/prabhas.jpeg')
        st.write('Prabhas')
    with cnt3:
        st.image('test_images/prakashraj.webp')
        st.write('Prakash')
    with cnt4:
        st.image('test_images/nagarjuna.jpg')
        st.write('Nagaruna')
  with st.container():
    cnt1,cnt2,cnt3,cnt4=st.columns(4)
    with cnt1:
        st.image('test_images/kajol.jpeg')
        st.write('Kajol')
    with cnt2:
        st.image('test_images/sridevi.webp')
        st.write('Sridevi')
    with cnt3:
        st.image('test_images/priyanka.jpeg')
        st.write('Priyanka')
    with cnt4:
        st.image('test_images/deepika.webp')
        st.write('Deepika')

elif choice=='Trained Actors':
    choice=st.selectbox('search for the actor that you want!!!',classes)
    if choice:
        st.write(f"Model is also trained for detecting {choice}. You can test with any image of {choice}.")
