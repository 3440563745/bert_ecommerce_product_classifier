
class TitlePredict:
    def __init__(self,predict):
        self.predict1 = predict
    def predict(self,text):
        return self.predict1.predict(text)