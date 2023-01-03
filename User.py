import dlib

class User:
    def __init__(self):
        self._rec = 0
        self._display_alert = None
        self._drowsiness_value = None
        self._drowsiness_val_submitted = None
        self._random_alert_frames = None
        self._panda_EAR = None
        self._EAR_data = None
        self._initial_EAR = None
        self._initial_mean = None
        self._initial_sd = None
        self._original_time = None
        self._Num_Frame = None
        # self._p = "shape_predictor_68_face_landmarks.dat"
        # self._detector = dlib.get_frontal_face_detector()
        # self._predictor = dlib.shape_predictor(self._p)
        # self._id=id

    # @property
    # def p(self):
    #     return self._p

    # @p.setter
    # def p(self, value):
    #     self._p = value

    # @property
    # def detector(self):
    #     return self._detector


    # @id.setter
    # def id(self, value):
    #     self._id = value

    # @property
    # def id(self):
    #     return self._id

    # @property
    # def predictor(self):
    #     return self._predictor

    # @detector.setter
    # def detector(self, color, value):
    #     rects = self._detector(color, value)
    #     return rects

    # @detector.setter
    # def predictor(self, color, rect):
    #     shape = self._predictor(color, rect)
    #     return shape

    @property
    def rec(self):
        return self._rec

    @rec.setter
    def rec(self, value):
        self._rec = value

    @property
    def display_alert(self):
        return self._display_alert

    @display_alert.setter
    def display_alert(self, value):
        self._display_alert = value

    @property
    def drowsiness_value(self):
        return self._drowsiness_value

    @drowsiness_value.setter
    def drowsiness_value(self, value):
        self._drowsiness_value = value

    @property
    def drowsiness_val_submitted(self):
        return self._drowsiness_val_submitted

    @drowsiness_val_submitted.setter
    def drowsiness_val_submitted(self, value):
        self._drowsiness_val_submitted = value

    @property
    def random_alert_frames(self):
        return self._random_alert_frames

    @random_alert_frames.setter
    def random_alert_frames(self, value):
        self._random_alert_frames = value

    @property
    def initial_EAR(self):
        return self._initial_EAR

    @initial_EAR.setter
    def initial_EAR(self, value):
        self._initial_EAR = value

    @property
    def EAR_data(self):
        return self._EAR_data

    @EAR_data.setter
    def EAR_data(self, value):
        self._EAR_data = value

    @property
    def initial_mean(self):
        return self._initial_mean

    @initial_mean.setter
    def initial_mean(self, value):
        self._initial_mean = value

    @property
    def panda_EAR(self):
        return self._panda_EAR

    @panda_EAR.setter
    def panda_EAR(self, value):
        self._panda_EAR = value

    @property
    def initial_sd(self):
        return self._initial_sd

    @initial_sd.setter
    def initial_sd(self, value):
        self._initial_sd = value

    @property
    def original_time(self):
        return self._original_time

    @original_time.setter
    def original_time(self, value):
        self._original_time = value

    @property
    def Num_Frame(self):
        return self._Num_Frame

    @Num_Frame.setter
    def Num_Frame(self, value):
        self._Num_Frame = value