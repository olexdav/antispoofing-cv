class Settings:
    def __init__(self):
        self.batch_size = 64
        self.device_name = "cuda"
    def get_params(self):
        return self.batch_size, self.device_name