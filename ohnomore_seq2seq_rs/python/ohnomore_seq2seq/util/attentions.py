from enum import Enum

class Attentions(Enum):
    luong = "luong"
    luong_monotonic = "luong_monotonic"
    bahdanau = "bahdanau"
    bahdanau_monotonic = "bahdanau_monotonic"