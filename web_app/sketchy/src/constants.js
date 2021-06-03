import { CircularProgress } from "@chakra-ui/react";
export const gray = "#F7FAFC";
export const black = "#000000";
export const white = "#FFFFFF";
export const darkGray = "#A3A8B0";
export const textColor = white;
export const backgroundColor = "#1A365D";
export const buttonHeight = "48px";
export const buttonWidth = "180px";
export const formLabelWidth = "200px";
export const custom_sketch_class = "My Custom Sketch"; // update python constant if modify
export const progress = (
  <CircularProgress
    isIndeterminate
    color={backgroundColor}
    size="180px"
    thickness="4px"
  />
);
export const colors = [
  "#e6194b",
  "#3cb44b",
  "#ffe119",
  "#4363d8",
  "#f58231",
  "#911eb4",
  "#46f0f0",
  "#f032e6",
  "#bcf60c",
  "#fabebe",
  "#008080",
  "#e6beff",
  "#9a6324",
  "#fffac8",
  "#800000",
  "#aaffc3",
  "#808000",
  "#ffd8b1",
  "#000075",
  "#808080",
  "#000000",
];
export const nb_to_show = 5;
export const datasets = ["Sketchy", "TU-Berlin", "Quickdraw"];

export const categories = {
  Quickdraw: [
    "airplane",
    "alarm_clock",
    "ant",
    "apple",
    "axe",
    "banana",
    "bandage",
    "bat",
    "beach",
    "bear",
    "bee",
    "bench",
    "bicycle",
    "bread",
    "bus",
    "butterfly",
    "cactus",
    "cake",
    "camel",
    "campfire",
    "candle",
    "car",
    "castle",
    "cat",
    "chair",
    "chandelier",
    "church",
    "couch",
    "cow",
    "crab",
    "crocodilian",
    "cruise ship",
    "cup",
    "dog",
    "dolphin",
    "door",
    "duck",
    "eiffel tower",
    "elephant",
    "eyeglasses",
    "fan",
    "feather",
    "fire_hydrant",
    "fish",
    "flower",
    "frog",
    "giraffe",
    "guitar",
    "hamburger",
    "hammer",
    "hat",
    "hedgehog",
    "helicopter",
    "horse",
    "hot-air_balloon",
    "hurricane",
    "kangaroo",
    "knife",
    "lighthouse",
    "lion",
    "lobster",
    "map",
    "megaphone",
    "monkey",
    "moon",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "owl",
    "palm tree",
    "parrot",
    "penguin",
    "pickup_truck",
    "pig",
    "pineapple",
    "pizza",
    "rabbit",
    "raccoon",
    "rhinoceros",
    "rifle",
    "sailboat",
    "saw",
    "saxophone",
    "scissors",
    "scorpion",
    "sea_turtle",
    "shark",
    "sheep",
    "shoe",
    "sink",
    "skateboard",
    "skyscraper",
    "snail",
    "snake",
    "soccer ball",
    "spider",
    "spoon",
    "squirrel",
    "swan",
    "teapot",
    "teddy_bear",
    "tiger",
    "train",
    "tree",
    "umbrella",
    "violin",
    "windmill",
    "wine bottle",
    "zebra",
  ],
  Sketchy: [
    "airplane",
    "alarm_clock",
    "ant",
    "ape",
    "apple",
    "armor",
    "axe",
    "banana",
    "bat",
    "bear",
    "bee",
    "beetle",
    "bell",
    "bench",
    "bicycle",
    "blimp",
    "bread",
    "butterfly",
    "cabin",
    "camel",
    "candle",
    "cannon",
    "car_(sedan)",
    "castle",
    "cat",
    "chair",
    "chicken",
    "church",
    "couch",
    "cow",
    "crab",
    "crocodilian",
    "cup",
    "deer",
    "dog",
    "dolphin",
    "door",
    "duck",
    "elephant",
    "eyeglasses",
    "fan",
    "fish",
    "flower",
    "frog",
    "geyser",
    "giraffe",
    "guitar",
    "hamburger",
    "hammer",
    "harp",
    "hat",
    "hedgehog",
    "helicopter",
    "hermit_crab",
    "horse",
    "hot_air_balloon",
    "hotdog",
    "hourglass",
    "jack_o_lantern",
    "jellyfish",
    "kangaroo",
    "knife",
    "lion",
    "lizard",
    "lobster",
    "motorcycle",
    "mouse",
    "mushroom",
    "owl",
    "parrot",
    "pear",
    "penguin",
    "piano",
    "pickup_truck",
    "pig",
    "pineapple",
    "pistol",
    "pizza",
    "pretzel",
    "rabbit",
    "raccoon",
    "racket",
    "ray",
    "rhinoceros",
    "rifle",
    "rocket",
    "sailboat",
    "saw",
    "saxophone",
    "scissors",
    "scorpion",
    "sea_turtle",
    "seagull",
    "seal",
    "shark",
    "sheep",
    "shoe",
    "skyscraper",
    "snail",
    "snake",
    "songbird",
    "spider",
    "spoon",
    "squirrel",
    "starfish",
    "strawberry",
    "swan",
    "sword",
    "table",
    "tank",
    "teapot",
    "teddy_bear",
    "tiger",
    "tree",
    "trumpet",
    "turtle",
    "umbrella",
    "violin",
    "volcano",
    "wading_bird",
    "wheelchair",
    "windmill",
    "window",
    "wine_bottle",
    "zebra",
  ],
  "TU-Berlin": [
    "airplane",
    "alarm_clock",
    "angel",
    "ant",
    "apple",
    "arm",
    "armchair",
    "ashtray",
    "axe",
    "backpack",
    "banana",
    "barn",
    "baseball_bat",
    "basket",
    "bathtub",
    "bear_(animal)",
    "bed",
    "bee",
    "beer_mug",
    "bell",
    "bench",
    "bicycle",
    "binoculars",
    "blimp",
    "book",
    "bookshelf",
    "boomerang",
    "bottle_opener",
    "bowl",
    "brain",
    "bread",
    "bridge",
    "bulldozer",
    "bus",
    "bush",
    "butterfly",
    "cabinet",
    "cactus",
    "cake",
    "calculator",
    "camel",
    "camera",
    "candle",
    "cannon",
    "canoe",
    "car_(sedan)",
    "carrot",
    "castle",
    "cat",
    "cell_phone",
    "chair",
    "chandelier",
    "church",
    "cigarette",
    "cloud",
    "comb",
    "computer_monitor",
    "computer_mouse",
    "couch",
    "cow",
    "crab",
    "crane_(machine)",
    "crocodile",
    "crown",
    "cup",
    "diamond",
    "dog",
    "dolphin",
    "donut",
    "door",
    "door_handle",
    "dragon",
    "duck",
    "ear",
    "elephant",
    "envelope",
    "eye",
    "eyeglasses",
    "face",
    "fan",
    "feather",
    "fire_hydrant",
    "fish",
    "flashlight",
    "floor_lamp",
    "flower_with_stem",
    "flying_bird",
    "flying_saucer",
    "foot",
    "fork",
    "frog",
    "frying_pan",
    "giraffe",
    "grapes",
    "grenade",
    "guitar",
    "hamburger",
    "hammer",
    "hand",
    "harp",
    "hat",
    "head",
    "head_phones",
    "hedgehog",
    "helicopter",
    "helmet",
    "horse",
    "hot_air_balloon",
    "hot_dog",
    "hourglass",
    "house",
    "human_skeleton",
    "ice_cream_cone",
    "ipod",
    "kangaroo",
    "key",
    "keyboard",
    "knife",
    "ladder",
    "laptop",
    "leaf",
    "lightbulb",
    "lighter",
    "lion",
    "lobster",
    "loudspeaker",
    "mailbox",
    "megaphone",
    "mermaid",
    "microphone",
    "microscope",
    "monkey",
    "moon",
    "mosquito",
    "motorbike",
    "mouse_(animal)",
    "mouth",
    "mug",
    "mushroom",
    "nose",
    "octopus",
    "owl",
    "palm_tree",
    "panda",
    "paper_clip",
    "parachute",
    "parking_meter",
    "parrot",
    "pear",
    "pen",
    "penguin",
    "person_sitting",
    "person_walking",
    "piano",
    "pickup_truck",
    "pig",
    "pigeon",
    "pineapple",
    "pipe_(for_smoking)",
    "pizza",
    "potted_plant",
    "power_outlet",
    "present",
    "pretzel",
    "pumpkin",
    "purse",
    "rabbit",
    "race_car",
    "radio",
    "rainbow",
    "revolver",
    "rifle",
    "rollerblades",
    "rooster",
    "sailboat",
    "santa_claus",
    "satellite",
    "satellite_dish",
    "saxophone",
    "scissors",
    "scorpion",
    "screwdriver",
    "sea_turtle",
    "seagull",
    "shark",
    "sheep",
    "ship",
    "shoe",
    "shovel",
    "skateboard",
    "skull",
    "skyscraper",
    "snail",
    "snake",
    "snowboard",
    "snowman",
    "socks",
    "space_shuttle",
    "speed_boat",
    "spider",
    "sponge_bob",
    "spoon",
    "squirrel",
    "standing_bird",
    "stapler",
    "strawberry",
    "streetlight",
    "submarine",
    "suitcase",
    "sun",
    "suv",
    "swan",
    "sword",
    "syringe",
    "t_shirt",
    "table",
    "tablelamp",
    "teacup",
    "teapot",
    "teddy_bear",
    "telephone",
    "tennis_racket",
    "tent",
    "tiger",
    "tire",
    "toilet",
    "tomato",
    "tooth",
    "toothbrush",
    "tractor",
    "traffic_light",
    "train",
    "tree",
    "trombone",
    "trousers",
    "truck",
    "trumpet",
    "tv",
    "umbrella",
    "van",
    "vase",
    "violin",
    "walkie_talkie",
    "wheel",
    "wheelbarrow",
    "windmill",
    "wine_bottle",
    "wineglass",
    "wrist_watch",
    "zebra",
  ],
};
