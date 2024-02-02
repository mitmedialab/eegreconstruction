#Creates a yaml file that maps selected ImageNet classes to their corresponding ids
from utils import write_yaml

image_ids = {
    "jacko_lantern": "n03590841",
    "banana": "n07753592",
    "airliner": "n02690373",
    "face": "n02917067",
    "panda": "n02510455",
    "pizza": "n07873807",
    "daisy": "n11939491",
    "anemone_fish": "n02606052",
    "tiger": "n02129604",
    "strawberry": "n07745940",
    "ambulance": "n02701002",
    "ice_cream": "n07615774",
    "coffee_mug": "n03063599",
    "electric_guitar": "n03272010",
    "basketball": "n02802426",
    "school_bus": "n04146614",
    "volcano": "n09472597",
    "red_wine": "n07892512",
    "beer": "n02823428",
    "castle": "n02980441",
}

write_yaml('./data/selected_image_ids.yaml', image_ids)