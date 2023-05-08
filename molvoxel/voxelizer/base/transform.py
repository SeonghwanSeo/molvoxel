from abc import ABCMeta, abstractmethod
class BaseRandomTransform(metaclass=ABCMeta) :
    def __init__(
        self,
        random_translation: float = 0.0,
        random_rotation: bool = False,
    ) :
        self.random_translation = random_translation
        self.random_rotation = random_rotation

    @abstractmethod
    def forward(self, coords, center):
        pass

    @abstractmethod
    def get_transform(self) :
        pass
