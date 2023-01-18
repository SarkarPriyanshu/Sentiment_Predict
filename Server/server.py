from Model.predict import predict
import streamlit as st

def main():
	# Predict class
	predict_ = predict()
	st.title('Sentiment Predictor APP')	
	text = st.text_input("Article for Sentiment Analysis : ","Paste text Here")
	result=""		
	if st.button("Predict"):
		result=predict_.predict_text(text)
		st.success('The output is {}'.format(result))

if __name__=="__main__":
	main()