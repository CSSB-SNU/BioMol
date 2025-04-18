class EntityMismatchError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class NonpolymerError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class AltIDError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
    
class StructConnAmbiguityError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class EmptyStructureError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)