import streamlit as st 
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

#titile
st.title("☘︎ Sentio")



# display the instructions
st.text("Classify comments as positive, negative or neutral by entering it in the text field")


def progress(rate,text,color):

	
	#this function creates progress bar based on the rate , text and  color
	

	

	progress_placeholder = st.empty()

	for percent_complete in range(rate):

		time.sleep(0.02) 

		progress_html = f"""
		<div style='width: 100%;'>
			<div style='margin-bottom: 5px; color: white; text-align: left; font-size:15px;'>{text} : {percent_complete + 1}%</div>
			<div style='width: 100%; background-color: rgb(38, 39, 48); border-radius: 100px;'>
				<div style='width: {percent_complete + 1}%; background-color: {color}; height: 6px; border-radius: 100px;'></div>
			</div>
			<div style='margin-bottom: 20px;margin-top: 30px;'></div> 
		</div>
		"""
		
		progress_placeholder.markdown(progress_html, unsafe_allow_html=True)


def dummy(text,color):

        #this function is create dummy progress bar , the initial animation is done using this function
 
	

	progress_placeholder = st.empty()

	for percent_complete in range(50):

		time.sleep(0.01) 
		
		progress_html = f"""
		<div style='width: 100%;'>
			<div style='margin-left:5px; margin-bottom: 5px; margin-top: 10px; color: white; text-align: left; font-size:15px ;'>{text} : {percent_complete + 1}%</div>
			<div style='margin-left:5px; width: 100%; background-color: rgb(38, 39, 48); border-radius: 100px;'>
				<div style='width: {percent_complete + 1}%; background-color: {color}; height: 6px; border-radius: 100px;'></div>
			</div>
			<div style='margin-bottom: 20px;margin-top: 30px;'></div>  
		</div>
		"""
		progress_placeholder.markdown(progress_html, unsafe_allow_html=True)

	for percent_complete in range(50, -1, -1):

		time.sleep(0.01)

		progress_html = f"""
		<div style='width: 100%;'>
			<div style='margin-left:5px; margin-bottom: 5px; margin-top: 10px; color: white; text-align: left; font-size:15px ;'>{text} : {percent_complete}%</div>
			<div style='margin-left:5px; width: 100%; background-color: rgb(38, 39, 48); border-radius: 100px;'>
				<div style='width: {percent_complete}%; background-color: {color}; height: 6px; border-radius: 100px;'></div>
			</div>
			<div style='margin-bottom: 20px;margin-top: 30px;'></div> 
		</div>
		"""
		progress_placeholder.markdown(progress_html, unsafe_allow_html=True)




#this is the form part 


with st.form(key='my_form'):

	theme = st.text_area(label='simply enter text')
	model = load_model('sentiment.h5')


	with open('tokenizer.pkl', 'rb') as f:
		tokenizer = pickle.load(f)

	text = [theme.lower()]
	text_seq = tokenizer.texts_to_sequences(text)
	text_padded = pad_sequences(text_seq, maxlen=200)


	prediction = model.predict(text_padded)

	submit_button = st.form_submit_button(label='Submit')
	

        #be careful using index to call value because prediction is a 2d array
        
	if theme and len(theme.strip()) > 0 and submit_button:
		progress(int(prediction[0][2]*100),"Positive","green")
		progress(int(prediction[0][1]*100),"Neutral","yellow")
		progress(int(prediction[0][0]*100),"Negative","red")
		value=np.argmax(prediction[0])
		if value==2:
			st.warning("This is a neutral comment 🙂")
		elif value==1:
			st.success("This is a positive comment 😊")
		else:
			st.error("This is a negative comment 😔")	
	else:
		dummy("Positive","green")
		dummy("Neutral","yellow")
		dummy("Negative","red")

#this is beginner friendly project if you want to implement this seriously add methods to handle exception and use class based structure rather than function based
#it is cleaner , reduce code there are many rooms for reducing the amount of code 



