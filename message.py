MessageType = {"init": 0, "track": 1, "result": 2}


class Message:
    def __init__(self, messageType, data):
        self.messageType = messageType
        self.data = data
