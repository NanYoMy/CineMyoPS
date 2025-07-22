class Sample():
    def __init__(self,img,lab):
        self.img=img
        self.lab=lab

class PathologySample():
    def __init__(self,img,myo_mask,lab):
        self.img=img
        self.myo_mask=myo_mask
        self.lab=lab