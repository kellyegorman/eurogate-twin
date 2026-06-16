class StraddleCarrier():
    def __init__(self, id):
        self.id = id

class Container():
    def __init__(self, id, size, weight, status, type, commodity):
        self.id = id
        self.size = size
        self.weight = weight
        self.status = status
        self.type = type
        self.commodity = commodity

        #change this later. We probably need information
        #like row, bay, and tier and maybe more.
        self.location = None

