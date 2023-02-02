import numpy as np

from typing import Any, List, Dict, Callable
from numpy.typing import ArrayLike

class ImageMaker() :
    def __init__(self, channels: List[str]) :
        self.channels = channels
        self.num_channels = len(channels)

    def split_channel(self, grids: ArrayLike) -> Dict[str, ArrayLike] :
        assert np.shape(grids)[0] == self.num_channels
        return {channel: grid for channel, grid in zip(self.channels, grids)}

class ChannelGetter() :
    def __init__(self, channels: List[str]) :
        self.channels = channels
        self.num_channels = len(channels)

class FeatureGetter(ChannelGetter) :
    CHANNEL_TYPE='FEATURE'
    def __init__(self, function: Callable[[Any], ArrayLike], channels: List[str]) :
        super().__init__(channels)
        self.feature_getter = function

    def get_feature(self, input: Any, **kwargs) -> ArrayLike :
        return self.feature_getter(input)

class TypeGetter(ChannelGetter) :
    CHANNEL_TYPE='TYPE'
    def __init__(self, types: List[Any], channels: List[str], unknown: bool = False) :
        if unknown :
            channels.append('Unknown')
        super().__init__(channels)
        type_dic = {typ: idx for idx, typ in enumerate(types)}
        if unknown :
            self.type_getter = lambda x: type_dic.get(x, self.num_channels-1)
        else :
            self.type_getter = lambda x: type_dic[x]
        
        def one_hot_encoding(idx) :
            res = [0] * self.num_channels
            res[idx] = 1
            return res
        self.feature_list = [one_hot_encoding(idx) for idx in range(self.num_channels)]

    def get_type(self, input: Any, **kwargs) -> int :
        return self.type_getter(input)

    def get_feature(self, input: Any, **kwargs) -> ArrayLike :
        return self.feature_list[self.get_type(input)]

    def to_feature_getter(self) :
        return FeatureGetter(self.feature_getter, self.channels)