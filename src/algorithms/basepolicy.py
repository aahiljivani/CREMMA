from abc import ABC, abstractmethod

class BasePolicy(ABC):
    @abstractmethod
    def predict(self, obs: np.ndarray, deterministic: bool)-> np.ndarray:
        ...
    @abstractmethod
    def update(self, replay_buffer, logger, step) -> None:
        ...

    @abstractmethod
    def save(self, path:str) -> None:
        ...

    @abstractmethod
    def load(self, path:str) -> None:
        ...

    