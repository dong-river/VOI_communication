PRICE_MAX_USD   = 300   # price_dollars = round(norm_price * PRICE_MAX_USD)
ARRIVAL_MAX_MIN = 180   # arrival_minutes = round(norm_arrival * ARRIVAL_MAX_MIN)
LAYOVER_MAX_MIN = 120   # longest_layover_minutes = round(norm_longest_stop * LAYOVER_MAX_MIN)
S_MAX_STOPS     = 2     # stops = round(norm_num_stops * S_MAX_STOPS)
RANDOM_SEED     = 8

FEATURE_ORDER = [
    "arrival_before_meeting", "american", "delta", "jetblue", "southwest",
    "longest_stop", "number_of_stops", "price"
]
AIRLINES = ["american", "delta", "jetblue", "southwest"]
AIRLINE_NAMES = {"american": "American", "delta": "Delta", "jetblue": "JetBlue", "southwest": "Southwest"}

def clamp01(x: float) -> float: return 0.0 if x < 0 else 1.0 if x > 1 else x
def d_price(x: float) -> int:   return int(round(clamp01(x) * PRICE_MAX_USD))
def d_arr(x: float) -> int:     return int(round(clamp01(x) * ARRIVAL_MAX_MIN))
def d_lay(x: float) -> int:     return int(round(clamp01(x) * LAYOVER_MAX_MIN))
def d_stops(x: float) -> int:   return int(round(clamp01(x) * S_MAX_STOPS))