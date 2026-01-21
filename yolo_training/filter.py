import os

# 1. Define the allowed suffixes (i.e. the parts after "ally_" or "enemy_")
allowed_suffixes = {
    "archer", "arrow", "baby_dragon", "brawler", "fireball", "giant",
    "goblin", "goblin_cage", "goblin_hut", "knight", "mini_pekka",
    "minion", "spear_goblin", "musketeer", "prince", "valkyrie"
}

# 2. Define the full list of class names as provided.
names = [
    'ally_archer', 'ally_arrow', 'ally_baby_dragon', 'ally_balloon', 'ally_bandit',
    'ally_barbarian', 'ally_bat', 'ally_battleram', 'ally_bomber', 'ally_bombtower',
    'ally_bowler', 'ally_brawler', 'ally_cursedhog', 'ally_dark_prince', 'ally_electro_wizard',
    'ally_elite_barbarian', 'ally_elixirpump', 'ally_executioner', 'ally_fire_spirit', 'ally_fireball',
    'ally_firecracker', 'ally_fisherman', 'ally_giant', 'ally_giant_bomb', 'ally_giant_skeleton',
    'ally_goblin', 'ally_goblin_barrel', 'ally_goblin_cage', 'ally_goblin_hut', 'ally_goblindrill',
    'ally_golem_mini', 'ally_graveyard', 'ally_guard', 'ally_hog_rider', 'ally_hungry_dragon',
    'ally_hunter', 'ally_ice_golem', 'ally_ice_spirit', 'ally_ice_wizard', 'ally_infernodragon',
    'ally_knight', 'ally_knight_evo', 'ally_log', 'ally_lumberjack', 'ally_magic_archer',
    'ally_mega_knight', 'ally_mega_minion', 'ally_mightyminer', 'ally_mini_pekka', 'ally_minion',
    'ally_mortar', 'ally_motherwitch', 'ally_musketeer', 'ally_pekka', 'ally_poison',
    'ally_prince', 'ally_princess', 'ally_rage', 'ally_rocket', 'ally_royale_giant',
    'ally_royalghost', 'ally_skeleton', 'ally_skeletonking', 'ally_spear_goblin', 'ally_tesla',
    'ally_tesla_hidden', 'ally_tombstone', 'ally_tornado', 'ally_valkyrie', 'ally_wallbreaker',
    'ally_witch', 'ally_wizard', 'ally_xbow', 'ally_zappy',
    'enemy_archer', 'enemy_arrow', 'enemy_baby_dragon', 'enemy_bandit', 'enemy_barbarian',
    'enemy_barbarian_evo', 'enemy_bat', 'enemy_bomb_tower', 'enemy_bomber', 'enemy_brawler',
    'enemy_dark_prince', 'enemy_dart_goblin', 'enemy_electro_wizard', 'enemy_electrogiant',
    'enemy_elixirgolem', 'enemy_executioner', 'enemy_fire_spirit', 'enemy_fireball',
    'enemy_firecracker', 'enemy_furnace', 'enemy_giant', 'enemy_giant_bomb', 'enemy_giant_skeleton',
    'enemy_goblin', 'enemy_goblin_barrel', 'enemy_goblin_cage', 'enemy_goblin_hut', 'enemy_goblinbarrel',
    'enemy_goldenknight', 'enemy_golem', 'enemy_golem_mini', 'enemy_guard', 'enemy_hog',
    'enemy_hog_rider', 'enemy_hunter', 'enemy_ice_golem', 'enemy_inferno_tower', 'enemy_infernodragon',
    'enemy_knight', 'enemy_log', 'enemy_lumberjack', 'enemy_magic_archer', 'enemy_mega_knight',
    'enemy_mega_minion', 'enemy_miner', 'enemy_mini_pekka', 'enemy_minion', 'enemy_musketeer',
    'enemy_night_witch', 'enemy_pekka', 'enemy_prince', 'enemy_princess', 'enemy_rage',
    'enemy_ramrider', 'enemy_rascalboy', 'enemy_rascalgirl', 'enemy_rocket', 'enemy_royal_ghost',
    'enemy_skeleton', 'enemy_skeleton_barrel', 'enemy_skeleton_evo', 'enemy_sparky',
    'enemy_spear_goblin', 'enemy_tesla', 'enemy_tesla_hidden', 'enemy_tombstone', 'enemy_valkyrie',
    'enemy_witch', 'enemy_wizard', 'enemy_xbow'
]

# 3. Compute the allowed indices based on your allowed suffixes.
allowed_indices = set()
for idx, classname in enumerate(names):
    # If the class starts with either "ally_" or "enemy_", check its suffix.
    if classname.startswith("ally_"):
        suffix = classname[len("ally_"):]
    elif classname.startswith("enemy_"):
        suffix = classname[len("enemy_"):]
    else:
        continue
    if suffix in allowed_suffixes:
        allowed_indices.add(idx)

print("Allowed class indices (from your list):", allowed_indices)

# 4. Set up your dataset directories.
# Adjust the base_path to the root folder of your dataset.
base_path = "dataset"  # <-- Change this to your dataset folder path
splits = ["train", "valid", "test"]  # <-- Adjust these if your splits differ

# List of common image file extensions.
image_extensions = [".jpg", ".png", ".jpeg"]

# Process each split.
for split in splits:
    labels_dir = os.path.join(base_path, split, "labels")
    images_dir = os.path.join(base_path, split, "images")

    # Get all .txt files in the labels directory.
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)

        with open(label_path, "r") as f:
            lines = f.readlines()

        # Check if any of the labels in the file is NOT allowed.
        remove_image = False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue  # Skip empty lines
            try:
                class_id = int(parts[0])
            except ValueError:
                # Skip lines that don't start with an integer.
                continue
            if class_id not in allowed_indices:
                remove_image = True
                break

        # If an unwanted label was found, remove the label file and its corresponding image.
        if remove_image:
            print(f"Removing {label_file} because it contains an unwanted class.")
            os.remove(label_path)

            # Match the label file with the corresponding image file (check by common extensions).
            base_filename = os.path.splitext(label_file)[0]
            for ext in image_extensions:
                image_file = base_filename + ext
                image_path = os.path.join(images_dir, image_file)
                if os.path.exists(image_path):
                    os.remove(image_path)
                    break

print("Dataset cleaning complete.")
