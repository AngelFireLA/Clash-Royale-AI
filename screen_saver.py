import concurrent.futures
import datetime
import json
import os
import random
import re
import time

import cv2
import dxcam
import easyocr
import keyboard
import numpy as np
import pyautogui
import win32api
from ultralytics import YOLO
from ultralytics.engine.results import Results

from window_controller import WindowController

controller = WindowController()


class Partie:
    def __init__(self):
        self.elixir_bleu = None
        self.elixir_rouge = None
        self.cartes_en_main = {0: None, 1: None, 2: None, 3: None}
        self.prochaine_carte = None
        self.tours_bleu = 3
        self.tours_rouge = 3
        self.pv_tours_rouge = {0: 1512, 1: 1512, 2: 2568}
        self.pv_tours_bleu = {3: 2352, 4: 2352, 5: 3768}
        self.position_tours_bleu = {3: (133, 654), 4: (422, 654), 5: (274, 713)}
        self.position_tours_rouge = [(133, 242), (422, 242), (274, 161)]
        self.elixir_timer_bleu = None
        self.elixir_timer_rouge = None
        self.chrono = None
        self.timer = None
        self.overtime = False
        self.blue_king_activated = False
        self.red_king_activated = False
        self.zone_placement_bleu = [((77, 480), (496, 750))]
        self.elixir_cooldown = 2.8


def find_image_in_screenshot(template, screenshot, return_coords=False, threshold=0.70):
    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val >= threshold:
        if return_coords:
            h, w = template.shape[:-1]

            center_x = int(max_loc[0] + w / 2)
            center_y = int(max_loc[1] + h / 2)
            return center_x, center_y
        return True
    return False


def get_side(x):
    if x < 960:
        return "left"
    else:
        return "right"


def distance(coords1, coords2):
    x1, y1 = coords1
    x2, y2 = coords2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def load_image(image_path):
    image = cv2.imread(image_path)
    return image


def current_time():
    time_now = datetime.datetime.now()
    time_now = time_now.strftime("%Y-%m-%d %H-%M-%S")
    return time_now


def get_elixir():
    # Define the color of empty squares and tolerance for matching
    empty_color = np.array([123, 54, 5])
    color_tolerance = 20

    # Define the list of x-coordinates to check
    x_coordinates = [183, 222, 261, 300, 339, 378, 417, 456, 495, 534]
    coordinates_to_check = [(x - 182, 993 - 980) for x in x_coordinates]
    # Capture the screen once
    screen = controller.screenshot()[980:1000, 182:550]

    def is_empty_square(px_color):
        for i in range(3):
            if abs(px_color[i] - empty_color[i]) > color_tolerance:
                return False
        return True

    # Count the number of non-full squares
    non_full_count = -1

    for x, y in coordinates_to_check:
        pixel_color = screen[y, x]
        if not is_empty_square(pixel_color):
            non_full_count += 1

    return non_full_count


def update_crowns():
    app_window_screenshot = controller.screenshot()[180:750, 45:490]
    # Find the location of the template image in the screenshot
    location1 = find_image_in_screenshot(blue_three_crown_image, app_window_screenshot)
    location2 = find_image_in_screenshot(red_three_crown_image, app_window_screenshot)

    if location1:
        return 2
    if location2:
        return 3

    blue_number_image = controller.screenshot()[534:564, 512:540]
    red_number_image = controller.screenshot()[336:365, 513:540]

    found = False
    images = numbers_images
    threshold = 0.80  # Adjust this threshold as needed

    for i, (image, number_image) in enumerate(zip(images, [blue_number_image] * 2 + [red_number_image] * 2)):
        result = cv2.matchTemplate(number_image, image, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)

        if loc[0].any():
            crown_number = 1 if i % 2 == 0 else 2
            if i < 2:  # Blue
                if partie.tours_rouge > 3 - crown_number:
                    partie.red_king_activated = True
                    partie.tours_rouge = 3 - crown_number
                    found = True
                    if partie.tours_rouge == 1:
                        partie.pv_tours_rouge[0] = None
                        partie.pv_tours_rouge[1] = None
                        partie.zone_placement_bleu = [((77, 351), (496, 750))]
                    break
            else:  # Red
                if partie.tours_bleu > 3 - crown_number:
                    partie.blue_king_activated = True
                    partie.tours_bleu = 3 - crown_number
                    found = True
                    if partie.tours_bleu == 1:
                        partie.pv_tours_bleu[3] = None
                        partie.pv_tours_bleu[4] = None
                    break

    if partie.tours_bleu == 2:
        tower_base_boxes = [((381, 618), (458, 685)), ((94, 618), (173, 685))]
        for box in tower_base_boxes:
            box_image = controller.screenshot()[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            coords = find_image_in_screenshot(broken_tower, box_image, return_coords=True)
            if coords:
                side = get_side(coords[0])
                if side == "left":
                    partie.pv_tours_bleu[3] = None
                else:
                    partie.pv_tours_bleu[4] = None

    if partie.tours_rouge == 2:
        tower_base_boxes = [((381, 218), (458, 282)), ((94, 218), (173, 282))]
        for box in tower_base_boxes:
            box_image = controller.screenshot()[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            coords = find_image_in_screenshot(broken_tower, box_image, return_coords=True)
            if coords:
                side = get_side(coords[0])
                if side == "left":
                    partie.pv_tours_rouge[0] = 0
                    partie.zone_placement_bleu = [((77, 480), (496, 750)), [(77, 480), (283, 750)]]
                else:
                    partie.pv_tours_rouge[1] = 0
                    partie.zone_placement_bleu = [((77, 480), (496, 750)), [(293, 480), (496, 750)]]

    return found and partie.overtime


def start_battle():
    app_window_screenshot = controller.screenshot()[102:173, 323:401]
    # Find the location of the template image in the screenshot
    location = find_image_in_screenshot(friends_icon, app_window_screenshot)

    if location:
        print("Friends icon found, starting battle...")
        time.sleep(0.1)
        # controller.click(app_coords[0] + (186), app_coords[1] + (679))

        controller.click(519, 139)
        time.sleep(0.1)
        controller.click(348, 349)
        time.sleep(0.1)
        controller.click(370, 599)

        location = find_image_in_screenshot(blason_de_combat, app_window_screenshot)
        while not location:
            app_window_screenshot = controller.screenshot()[375:550, 175:375]
            # Find the location of the template image in the screenshot
            location = find_image_in_screenshot(blason_de_combat, app_window_screenshot)

        print("battle crest found")
        time.sleep(3)
        partie.chrono = 175
        partie.timer = time.time()
        elixir = get_elixir()
        if elixir:
            partie.elixir_bleu = elixir
            print(f"detected {elixir} elixir")
            partie.elixir_timer_bleu = time.time()
            partie.elixir_rouge = elixir
            partie.elixir_timer_rouge = time.time()
        else:
            raise ValueError("Elixir not found")
        return True
    else:
        print("friends not found")
        return False


def exit_battle():
    app_window_screenshot = controller.screenshot()[827:904, 16:88]

    # Find the location of the template image in the screenshot
    location = find_image_in_screenshot(exit_battle_red_cross_button, app_window_screenshot)

    if location:
        print("Exiting battle...")
        time.sleep(0.1)
        controller.click(54, 866)
        time.sleep(0.1)
        controller.click(384, 636)
        time.sleep(7)
        controller.click(270, 870)
        image_to_find = friends_icon
        app_window_screenshot = controller.screenshot()[102:173, 323:401]

        # Find the location of the template image in the screenshot
        location = find_image_in_screenshot(image_to_find, app_window_screenshot)
        while not location:
            app_window_screenshot = controller.screenshot()[102:173, 323:401]
            # Find the location of the template image in the screenshot
            location = find_image_in_screenshot(image_to_find, app_window_screenshot)


def find_troops(screenshot):
    results: Results = model.predict(source=screenshot, stream=True, conf=0.4, verbose=False, device='gpu')
    for result in results:
        detections = []
        results_json = result.to_json()
        try:
            results_data = json.loads(results_json)
        except json.JSONDecodeError:
            print("Error decoding JSON:", results_json)
            return []

        for item in results_data:
            box = item.get('box')
            if not box:
                continue

            try:
                center_x = (box['x1'] + box['x2']) / 2
                center_y = (box['y1'] + box['y2']) / 2
            except (KeyError, TypeError):
                print("Error processing box data:", box)
                continue

            detections.append((item['name'], (center_x, center_y)))
        return detections, result


def clean_string(input_string):
    # Use regular expression to remove anything that is not a number or letter
    cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    cleaned_string = cleaned_string.replace("G", "6")
    cleaned_string = cleaned_string.replace("S", "5")
    cleaned_string = cleaned_string.replace("t", "1")
    cleaned_string = cleaned_string.replace("T", "1")
    cleaned_string = cleaned_string.replace("e", "2")
    cleaned_string = cleaned_string.replace("q", "4")
    cleaned_string = cleaned_string.replace("q", "2")
    cleaned_string = cleaned_string.replace(" ", "")
    return cleaned_string


def parse_image(i, coord):
    try:
        if i < 3 and not partie.pv_tours_rouge[i]:
            return None, None
        elif i >= 3 and not partie.pv_tours_bleu[i]:
            return None, None

        img = controller.screenshot()[coord[0][1]:coord[1][1], coord[0][0]:coord[1][0]]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresholded = cv2.bitwise_not(gray)

        text = reader.readtext(thresholded)[0][1]

        # text = clean_string(text)

        if text == '' or text == " ":
            return None, None

        hp = int(text)
        if i < 3:
            if partie.pv_tours_rouge[i] and hp < partie.pv_tours_rouge[i] and partie.pv_tours_rouge[i] - hp < 1100:
                return i, hp
        else:
            if partie.pv_tours_bleu[i] and hp < partie.pv_tours_bleu[i] and partie.pv_tours_bleu[i] - hp < 1100:
                return i, hp
    except:
        pass
    return None, None


def filter_not_none(item):
    i, text = item
    if i is not None and text is not None:
        return True
    return False


executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def detect_tower_hp():
    coords = [((397, 166), (436, 190)), ((106, 166), (145, 190)), ((275, 46), (331, 69)),
              ((397, 640), (449, 660)), ((106, 640), (159, 660)), ((275, 777), (329, 796))]
    coords = [((x1, y1), (x2, y2)) for (x1, y1), (x2, y2) in coords]

    parse_results = list(executor.map(parse_image, range(len(coords)), coords))

    for i, text in filter(filter_not_none, parse_results):
        if i < 3:
            if text:
                partie.pv_tours_rouge[i] = text
        else:
            if text:
                partie.pv_tours_bleu[i] = text


def place_card(card_slot_index, pos):
    global time_since_last_troop
    time_since_last_troop = time.time()
    controller.click(card_slots_positions[card_slot_index][0], card_slots_positions[card_slot_index][1])
    time.sleep(0.01)
    controller.click(pos[0], pos[1])
    partie.elixir_bleu -= all_cards_cost[partie.cartes_en_main[card_slot_index]]
    partie.cartes_en_main[card_slot_index] = None


def get_current_cards():
    screenshot = controller.screenshot()[card_slot_rectangles[0][1]:card_slot_rectangles[1][1], card_slot_rectangles[0][0]:card_slot_rectangles[1][0]]
    for i in range(4):
        x1 = card_slot_boxes[i][0][0] - card_slot_rectangles[0][0]
        y1 = card_slot_boxes[i][0][1] - card_slot_rectangles[0][1]
        x2 = card_slot_boxes[i][1][0] - card_slot_rectangles[0][0]
        y2 = card_slot_boxes[i][1][1] - card_slot_rectangles[0][1]
        card_slot = screenshot[y1:y2, x1:x2]
        for card, card_image in preloaded_card_icons.items():
            if find_image_in_screenshot(card_image, card_slot):
                partie.cartes_en_main[i] = card
                break

    # next_card_slot = take_screenshot(rectangle_corners=next_card_slot_box)
    # for card, card_image in preloaded_card_icons.items():
    #     if find_image_in_screenshot(card_image, next_card_slot):
    #         partie.prochaine_carte = card
    #         break


def find_optimal_center(coordinates, radius):
    max_count = 0
    max_center = None

    for potential_center in coordinates:
        count = 0
        for other_point in coordinates:
            if distance(potential_center, other_point) <= radius:
                count += 1
        if count > max_count:
            max_count = count
            max_center = potential_center

    return max_center, max_count


def blue_tower_index(side):
    if side == "left":
        return 3
    elif side == "right":
        return 4
    else:
        raise ValueError("Invalid side")


log_strategy = False


def bot(enemy_troops, ally_troops):
    cards_in_hand = list(partie.cartes_en_main.values())
    # print("Cards in hand :", cards_in_hand)
    # print("Elixir :", partie.elixir_bleu)
    current_win_conditions = [element for element in cards_in_hand if element in win_conditions]
    current_cycle_cards = [element for element in cards_in_hand if element in cycle_cards]
    current_building_cards = [element for element in cards_in_hand if element in buildings]
    current_support_cards = [element for element in cards_in_hand if element in support_cards]
    arrow_damage = 93
    target_tower = random.choice(["left", "right"])
    if partie.tours_rouge == 3:
        game_phase = "start"
        enemies_left = len(enemy_troops["left"][0])
        enemies_right = len(enemy_troops["right"][0])
        if partie.pv_tours_rouge[1] < partie.pv_tours_rouge[0]:
            if abs(partie.pv_tours_rouge[1] - partie.pv_tours_rouge[0]) >= 500:
                target_tower = "left"
            else:
                if enemies_left <= enemies_right:
                    target_tower = "left"
                else:
                    target_tower = "right"
        elif partie.pv_tours_rouge[1] > partie.pv_tours_rouge[0]:
            if abs(partie.pv_tours_rouge[1] - partie.pv_tours_rouge[0]) >= 500:
                target_tower = "right"
            else:
                if enemies_right <= enemies_left:
                    target_tower = "right"
                else:
                    target_tower = "left"
        else:
            if enemies_right < enemies_left:
                target_tower = "right"
            elif enemies_right > enemies_left:
                target_tower = "left"
            else:
                target_tower = random.choice(["left", "right"])

    elif partie.tours_rouge == 2:
        game_phase = "end"
        if partie.pv_tours_rouge[1]:
            target_tower = "right"
        elif partie.pv_tours_rouge[0]:
            target_tower = "left"
    else:
        game_phase = "middle"

    # print("Chosen side :", target_tower)

    if log_strategy: print("Elixir :", partie.elixir_bleu)
    if log_strategy: print("Current win conditions :", win_conditions, "Current support cards :", current_support_cards,
                           "Current cycle cards :", current_cycle_cards, "Current building cards :",
                           current_building_cards)
    if game_phase:  # pour l'instant, on ne vérifie pas la game phase, car ça ne sert à rien vu la complexité du bot
        plan = None
        attacking_enemy_troops = [troop for troop in enemy_troops["all"][0] if
                                  enemy_troops["all"][1][enemy_troops["all"][0].index(troop)][1] >
                                  BLUE_BRIDGE_COORDS["left"][1]]
        if attacking_enemy_troops:
            plan = "defend"
            if log_strategy: print("Mode: Defense")
        else:
            plan = "attack"
            if log_strategy: print("Mode: Attack")
            if log_strategy and attacking_enemy_troops: print("enemy troops : ", attacking_enemy_troops)

        # do for any plan
        if plan:
            if "ally_arrow" in cards_in_hand and partie.elixir_bleu >= all_cards_cost["ally_arrow"] and len(
                    enemy_troops["all"][0]) >= 3:
                optimal_center, count = find_optimal_center(enemy_troops["all"][1], int(2 * TILE_SIZE))
                if count >= 3:
                    place_card(cards_in_hand.index("ally_arrow"),
                               (optimal_center[0], optimal_center[1] + TILE_SIZE))
                    if log_strategy: print(f"Found cluster of {count} troops in {optimal_center}, playing arrows.")

        # do for attack plan
        if plan == "attack":
            if partie.elixir_bleu >= 8:
                if current_win_conditions and current_support_cards:
                    place_card(cards_in_hand.index(current_win_conditions[0]), BLUE_BRIDGE_COORDS[target_tower])
                    place_card(cards_in_hand.index(current_support_cards[0]),
                               (BLUE_BRIDGE_COORDS[target_tower][0], BLUE_BRIDGE_COORDS[target_tower][1] + 60))
                    print(
                        f"Placing combo {current_win_conditions[0].split('ally_')[1]} and {current_support_cards[0].split('ally_')[1]} to push.")
                elif current_cycle_cards:
                    place_card(cards_in_hand.index(current_cycle_cards[0]), BLUE_BACK_COORDS[target_tower])
                    print(f"Placing {current_cycle_cards[0].split('ally_')[1]} to cycle.")
                elif partie.elixir_bleu == 9:
                    if "ally_mini_pekka" in partie.cartes_en_main and "ally_prince" in partie.cartes_en_main:
                        place_card(cards_in_hand.index("ally_prince"), BLUE_BRIDGE_COORDS[target_tower])
                        place_card(cards_in_hand.index("ally_mini_pekka"),
                                   (BLUE_BRIDGE_COORDS[target_tower][0], BLUE_BRIDGE_COORDS[target_tower][1] + 60))
                        print("Placing combo mini pekka prince")

            elif partie.elixir_bleu >= 3:
                if target_tower == "left" and partie.pv_tours_rouge[0] and partie.pv_tours_rouge[0] <= arrow_damage * 3:
                    if "ally_arrow" in cards_in_hand:
                        place_card(cards_in_hand.index("ally_arrow"), RED_TOWER_COORDS[target_tower])
                        print(f"Placing arrows on the left tower with hp remaining {partie.pv_tours_rouge[0]}")
                elif target_tower == "right" and partie.pv_tours_rouge[1] and partie.pv_tours_rouge[1] <= arrow_damage * 3:
                    if "ally_arrow" in cards_in_hand:
                        place_card(cards_in_hand.index("ally_arrow"), RED_TOWER_COORDS[target_tower])
                        print("Placing arrows on the right tower with hp remaining", partie.pv_tours_rouge[1])
                elif partie.pv_tours_rouge[2] and partie.pv_tours_rouge[2] <= arrow_damage * 3:
                    if "ally_arrow" in cards_in_hand:
                        place_card(cards_in_hand.index("ally_arrow"), RED_TOWER_COORDS[target_tower])
                        print("Placing arrows on the king tower with hp remaining", partie.pv_tours_rouge[2])

        # do for defend plan
        elif plan == "defend":

            troop_to_defend = None
            closest_distance = None
            closest_troop_coord = None
            for troop in attacking_enemy_troops:
                troop_coord = enemy_troops["all"][1][enemy_troops["all"][0].index(troop)]
                tower_distance = distance(partie.position_tours_bleu[blue_tower_index(get_side(troop_coord[0]))],
                                          troop_coord)
                if closest_distance is not None:
                    if tower_distance < closest_distance:
                        troop_to_defend = troop
                        closest_distance = tower_distance
                        closest_troop_coord = troop_coord
                else:
                    troop_to_defend = troop
                    closest_distance = tower_distance
                    closest_troop_coord = troop_coord

            if log_strategy: print(troop_to_defend, "is attacking")
            pos_enemy = closest_troop_coord
            no_one_already_defending = True
            try:
                goblin_cage_defending = "ally_gobelin_cage" in ally_troops[get_side(pos_enemy[0])][0]
                mini_pekka_defending = "ally_mini_pekka" in ally_troops[get_side(pos_enemy[0])][0]
            except Exception as e:
                print(ally_troops)
                print(ally_troops[get_side(pos_enemy[0])])
                print(pos_enemy[0])
                raise e
            if mini_pekka_defending:
                mini_pekka_defending_pos = ally_troops[get_side(pos_enemy[0])][1][
                    ally_troops[get_side(pos_enemy[0])][0].index("ally_mini_pekka")]
                if distance(mini_pekka_defending_pos, pos_enemy) < TILE_SIZE * 7:
                    if log_strategy: print("mini pekka already defending")
                    no_one_already_defending = False
            if goblin_cage_defending:
                if log_strategy: print("goblin cage already defending")
                no_one_already_defending = False

            if no_one_already_defending:
                if distance(partie.position_tours_bleu[blue_tower_index(get_side(pos_enemy[0]))],
                            pos_enemy) > TILE_SIZE * 7 and "ally_goblin_cage" in cards_in_hand and partie.elixir_bleu >= 4 and (
                        "enemy_giant" in enemy_troops["all"][0] or "enemy_prince" in enemy_troops["all"][0] or "enemy_mini_pekka" in enemy_troops["all"][0] or "enemy_knight" in enemy_troops["all"][0]):
                    place_card(cards_in_hand.index("ally_goblin_cage"), BLUE_MIDDLE_COORDS)
                    if log_strategy: print(f"Defending with {pos_enemy} with Goblin Cage")
                else:
                    if "ally_mini_pekka" in cards_in_hand and partie.elixir_bleu >= 4 and (
                            "enemy_giant" in enemy_troops["all"][0] or "enemy_prince" in enemy_troops["all"][0]):
                        place_card(cards_in_hand.index("ally_mini_pekka"),
                                   (pos_enemy[0], pos_enemy[1] + int(TILE_SIZE * 2)))
                        if log_strategy: print("Defending Giant with Mini Pekka")
                    elif "ally_mini_pekka" in cards_in_hand and partie.elixir_bleu >= 4 and "enemy_mini_pekka" in \
                            enemy_troops["all"][0]:
                        place_card(cards_in_hand.index("ally_mini_pekka"),
                                   (pos_enemy[0] + int(TILE_SIZE * 2), pos_enemy[1] + int(TILE_SIZE * 2)))
                    elif "ally_goblin" in cards_in_hand and partie.elixir_bleu >= 2:
                        place_card(cards_in_hand.index("ally_goblin"),
                                   (pos_enemy[0], pos_enemy[1] + int(TILE_SIZE * 2)))
                    elif "ally_knight" in cards_in_hand and partie.elixir_bleu >= 3:
                        place_card(cards_in_hand.index("ally_knight"),
                                   (pos_enemy[0], pos_enemy[1] + int(TILE_SIZE * 2)))
                    elif "ally_archer" in cards_in_hand and partie.elixir_bleu >= 3:
                        place_card(cards_in_hand.index("ally_archer"),
                                   (pos_enemy[0] + int(TILE_SIZE * 2), pos_enemy[1]))
                    elif "ally_prince" in cards_in_hand and partie.elixir_bleu >= 5:
                        place_card(cards_in_hand.index("ally_prince"),
                                   (pos_enemy[0] + int(TILE_SIZE * 2), pos_enemy[1]))


TILE_SIZE = 26
model = YOLO("epoch50.pt")
partie = Partie()

numbers_images = [cv2.imread("images/battle/un_bleu.png"), cv2.imread("images/battle/deux_bleu.png"),
                  cv2.imread("images/battle/un_rouge.png"), cv2.imread("images/battle/deux_rouge.png")]
numbers_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in numbers_images]
card_slots_positions = [(177, 913), (277, 913),
                        (377, 913), (477, 913)]

card_slot_boxes = [((125, 848), (221, 971)),
                   ((229, 848), (325, 971)),
                   ((333, 848), (429, 971)),
                   ((437, 848), (533, 971))]
card_slot_rectangles = ((120, 840), (540, 980))
next_card_slot_box = ((25, 948), (74, 1008))
all_cards = ["ally_knight", "ally_archer", "ally_prince", "ally_goblin",
             "ally_spear_goblin", "ally_giant",
             "ally_mini_pekka", "ally_minion", "ally_arrow", "ally_fireball",
             "ally_goblin_cage", "ally_musketeer",
             "ally_goblin_hut"]
threat_level = {
    "enemy_spear_goblin": 1,
    "enemy_goblin": 1,
    "enemy_minion": 2,
    "enemy_archer": 2,
    "enemy_knight": 3,
    "enemy_musketeer": 3,
    "enemy_mini_pekka": 4,
    "enemy_giant": 5,
    "enemy_prince": 5,
}
defending_level = {
    "ally_archer": 1,
    "ally_spear_goblin": 1,
    "ally_goblin": 2,
    "ally_minion": 2,
    "ally_musketeer": 2,
    "ally_goblin_hut": 2,
    "ally_knight": 3,
    "ally_goblin_cage": 3,
    "ally_mini_pekka": 4,
    "ally_prince": 5,
}
main_attack_level = {
    "ally_goblin": 1,
    "ally_knight": 3,
    "ally_mini_pekka": 4,
    "ally_prince": 5,
    "ally_giant": 5,
}
main_support_level = {
    "ally_giant": 0,
    "ally_knight": 1,
    "ally_spear_goblin": 2,
    "ally_goblin": 2,
    "ally_minion": 2,
    "ally_archer": 3,
    "ally_musketeer": 3,
    "ally_mini_pekka": 4,
    "ally_prince": 4,
}
cycle_level = {
    "ally_giant": 1,
    "ally_prince": 1,
    "ally_musketeer": 2,
    "ally_mini_pekka": 2,
    "ally_goblin_cage": 2,
    "ally_goblin_hut": 2,
    "ally_minion": 3,
    "ally_goblin": 3,
    "ally_spear_goblin": 4,
    "ally_knight": 5,
    "ally_archer": 5,
}

all_cards_cost = {
    "ally_knight": 3,
    "ally_archer": 3,
    "ally_prince": 5,
    "ally_goblin": 2,
    "ally_spear_goblin": 2,
    "ally_giant": 5,
    "ally_mini_pekka": 4,
    "ally_minion": 3,
    "ally_arrow": 3,
    "ally_fireball": 4,
    "ally_goblin_cage": 4,
    "ally_musketeer": 4,
    "ally_goblin_hut": 5
}
deck = ["ally_prince", "ally_mini_pekka", "ally_archer", "ally_knight", "ally_giant", "ally_goblin_cage", "ally_minion",
        "ally_goblin"]
win_conditions = ["ally_prince", "ally_giant", "ally_mini_pekka"]
cycle_cards = ["ally_archer", "ally_spear_goblin", "ally_goblin", "ally_knight", "ally_minion"]
buildings = ["ally_goblin_cage", "ally_goblin_hut"]
support_cards = ["ally_archer", "ally_minion", "ally_spear_goblin", "ally_goblin"]
building_targetting = ["enemy_giant"]

blue_three_crown_image = load_image('images/battle/three_crown_blue.png')
red_three_crown_image = load_image('images/battle/three_crown_red.png')
friends_icon = load_image('images/battle/friends_icon.png')
blason_de_combat = load_image('images/battle/vs_blason_debut_de_combat.png')
exit_battle_red_cross_button = load_image('images/battle/exit_battle_red_cross_button.png')
broken_tower = load_image('images/battle/broken_tower.png')

min_time_between_troops = 0.5
time_since_last_troop = time.time()
BLUE_MIDDLE_COORDS = (265, 560)
BLUE_BRIDGE_COORDS = {"left": (128, 438), "right": (419, 457)}
BLUE_BACK_COORDS = {"left": (47, 750), "right": (502, 751)}
# red tower left :( 135 ) ( 252 )
# red tower right : ( 423 ) ( 248 )
RED_TOWER_COORDS = {"left": (135, 252), "right": (423, 248)}
preloaded_card_icons = {card: load_image(f"images/cards/{card}.png") for card in deck}
camera = dxcam.create(output_color="BGR")
reader = easyocr.Reader(['en'])
screen_width, screen_height = pyautogui.size()
print(f"Ready!= ")


def start():
    mode = None
    timer = None
    recording = False
    record_number = 0
    live_detection = False
    start_time = None
    fps_start_time = time.time()
    time_since_last_recording = time.time()
    frame_count = 0
    fps_values = []
    while True:
        # 1 exit
        if keyboard.is_pressed('&'):  # 1
            exit()

        # 2 screenshot
        elif keyboard.is_pressed('é'):  # 2
            ss_mode = "multiple"
            if ss_mode == "single":
                current_date = current_time()

                start_time = time.time()

                # Adjust the region to exclude the toolbar
                app_window_screenshot = controller.screenshot()
                # Convert the color channels from BGR to RGB
                cv2.imwrite(f"images/single_ss/{current_date}.png", app_window_screenshot)

                print(f"Screenshot taken and saved in {time.time() - start_time} seconds")
                print()
                mode = None
                time.sleep(0.5)
            if ss_mode == "multiple":
                coords = [((401, 162), (450, 185)), ((111, 162), (155, 185)), ((282, 46), (331, 69)),
                          ((401, 644), (450, 662)), ((111, 644), (155, 662)), ((279, 777), (329, 796))]
                coords = [((x1, y1), (x2, y2)) for (x1, y1), (x2, y2)
                          in coords]

                for coord in coords:
                    current_date = current_time()

                    start_time = time.time()

                    # Adjust the region to exclude the toolbar
                    app_window_screenshot = controller.screenshot()[coord[0][1]:coord[1][1], coord[0][0]:coord[1][0]]
                    # Convert the color channels from BGR to RGB
                    cv2.imwrite(f"images/single_ss/{current_date}_{coord[0][0]}.png", app_window_screenshot)

                    print(f"Screenshot taken and saved in {time.time() - start_time} seconds")
                    print()
                    mode = None
                    time.sleep(0.1)

        # 3 Start battle
        elif mode != "battle" and keyboard.is_pressed('"'):  # 3
            mode = "start_battle"
            print("starting battle")
        # 4 Force Start battle
        elif mode != "battle" and keyboard.is_pressed("'"):  # 4
            mode = "force_start_battle"

        # 5 toggle recording
        elif keyboard.is_pressed("("):  # 5
            if recording:
                recording = False
                print("stopped recording")
                time.sleep(0.5)
            else:
                recording = True
                record_number = 0
                recording_date = current_time()
                print("started recording")
                # create folder of the date:
                if not os.path.exists(f"images/recording/{recording_date}"):
                    os.makedirs(f"images/recording/{recording_date}")
                time.sleep(0.1)

        # 6 toggle live detection
        elif keyboard.is_pressed("-"):
            if live_detection:
                live_detection = False
                print("stopped detecting")
                time.sleep(0.1)
            else:
                live_detection = True
                print("started detecting")
                time.sleep(0.1)

        if mode != "battle" and win32api.GetKeyState(0x04) < 0:
            if timer:
                print(time.time() - timer)

                timer = None
            else:
                timer = time.time()
                print("timer started")

            # x,y = pyautogui.position().x, pyautogui.position().y
            # pixel_color = pyautogui.pixel(x, y)
            # print(f"RGB Color at ({x}, {y}): {pixel_color}")

            time.sleep(0.1)

        elif mode != "battle" and (mode == "start_battle" or mode == "force_start_battle"):
            if start_battle() or mode == "force_start_battle":
                if mode == "force_start_battle":
                    partie.chrono = 175
                    partie.timer = time.time()
                    elixir = get_elixir()
                    if elixir:
                        partie.elixir_bleu = elixir
                        partie.elixir_timer_bleu = time.time()
                        partie.elixir_rouge = elixir
                        partie.elixir_timer_rouge = time.time()
                mode = "battle"
                print("Battle started")

            else:
                mode = None
                time.sleep(0.1)
            # start_recording()
            # record_number = 0
            # current_date = current_time()
            # print("started recording")

        if mode == "battle":

            start = start_time = time.time()
            partie.elixir_bleu = get_elixir()
            get_elixir_time = time.time() - start
            # print()
            # print("Loop")
            # 1. Finding troops
            start = time.time()
            screenshot = controller.screenshot()
            found_troops = find_troops(screenshot)

            find_troops_time = time.time() - start

            all_troops = found_troops[0]
            troops = {"left": ([], []), "right": ([], []), "all": ([], [])}
            ally_troops = {"left": ([], []), "right": ([], []), "all": ([], [])}
            for troop in all_troops:
                if troop[0].startswith("enemy_"):
                    if troop[1][1] <= int(540 / 2):
                        troops["left"][0].append(troop[0])
                        troops["left"][1].append(troop[1])
                    else:
                        troops["right"][0].append(troop[0])
                        troops["right"][1].append(troop[1])
                    troops["all"][0].append(troop[0])
                    troops["all"][1].append(troop[1])
                if troop[0].startswith("ally_"):
                    if troop[1][1] <= int(540 / 2):
                        ally_troops["left"][0].append(troop[0])
                        ally_troops["left"][1].append(troop[1])
                    else:
                        ally_troops["right"][0].append(troop[0])
                        ally_troops["right"][1].append(troop[1])
                    ally_troops["all"][0].append(troop[0])
                    ally_troops["all"][1].append(troop[1])

            # if partie.elixir_bleu > 10:
            #     partie.elixir_bleu = 10
            # if partie.elixir_rouge > 10:
            #     partie.elixir_rouge = 10
            # if partie.elixir_bleu < 0:
            #     raise ValueError("Blue Elixir can't be negative ?")
            # if partie.elixir_rouge < 0:
            #     raise ValueError("Red Elixir can't be negative ?")
            # if partie.elixir_bleu < 10 and time.time() - partie.elixir_timer_bleu >= partie.elixir_cooldown:
            #     partie.elixir_bleu += (time.time() - partie.elixir_timer_bleu)//partie.elixir_cooldown
            #     partie.elixir_timer_bleu = time.time() - ((time.time() - partie.elixir_timer_bleu) % partie.elixir_cooldown)+0.2
            # elif partie.elixir_bleu >= 10:
            #     partie.elixir_timer_bleu = time.time()+0.2
            # if partie.elixir_rouge < 10 and time.time() - partie.elixir_timer_rouge >= partie.elixir_cooldown:
            #     partie.elixir_rouge += (time.time() - partie.elixir_timer_rouge)//partie.elixir_cooldown
            #     partie.elixir_timer_rouge = time.time() - ((time.time() - partie.elixir_timer_rouge) % partie.elixir_cooldown)+0.2
            # elif partie.elixir_rouge >= 10:
            #     partie.elixir_timer_rouge = time.time()+0.2

            start = time.time()
            game_ended_status = update_crowns()
            update_crowns_time = time.time() - start
            # print(partie.tours_bleu, partie.tours_rouge)
            if game_ended_status == 2:
                print("Game ended with blue victory")
                mode = None
                continue
            if game_ended_status == 3:
                print("Game ended with red victory")
                mode = None
                continue

            if game_ended_status:
                print("blue", partie.tours_bleu, "red", partie.tours_rouge)
                if partie.tours_bleu == partie.tours_rouge:
                    print("Game ended by draw")
                    mode = None
                    continue
                elif partie.tours_bleu > partie.tours_rouge:
                    print("Game ended with blue victory")
                    mode = None
                    continue
                elif partie.tours_bleu < partie.tours_rouge:
                    print("Game ended with red victory")
                    mode = None
                    continue

            if time.time() - partie.timer >= 1:
                partie.timer = time.time() - ((time.time() - partie.timer) % 1)
                partie.chrono -= (time.time() - partie.timer) // 1
                egalite = partie.tours_bleu == partie.tours_rouge
                if partie.chrono <= 60:
                    if partie.overtime:
                        partie.elixir_cooldown = 0.9
                    else:
                        partie.elixir_cooldown = 1.4
                elif partie.chrono <= 0:
                    if not partie.overtime and egalite:
                        print("overtime")
                        partie.overtime = True
                        partie.chrono = 120
                    else:
                        partie.overtime = False
                        if partie.tours_bleu == partie.tours_rouge:
                            print("Game ended with draw")
                            mode = None
                            continue
                        elif partie.tours_bleu > partie.tours_rouge:
                            print("Game ended with blue victory")
                            mode = None
                            continue
                        elif partie.tours_bleu < partie.tours_rouge:
                            print("Game ended with red victory")
                            mode = None
                            continue

                # print(partie.chrono)

            start = time.time()
            detect_tower_hp()
            detect_tower_hp_time = time.time() - start
            start = time.time()
            # if there's a None in the cards in hand :
            if None in partie.cartes_en_main.values() or True:
                get_current_cards()
                get_current_cards_time = time.time() - start

            if start_time - time_since_last_troop >= min_time_between_troops:
                bot(troops, ally_troops)

            # # Log the performance times
            # print(f"find_troops: {find_troops_time:.4f} s")
            # print(f"get_elixir: {get_elixir_time:.4f} s")
            # print(f"update_crowns: {update_crowns_time:.4f} s")
            # print(f"detect_tower_hp: {detect_tower_hp_time:.4f} s")
            # print(f"get_current_cards: {get_current_cards_time:.4f} s")
            # #print total time
            # print(f"Total time: {time.time() - start_time:.4f} s")
            # FPS Counter
            frame_count += 1
            if time.time() - fps_start_time >= 1:
                fps = frame_count / (time.time() - fps_start_time)
                if fps > 4:
                    fps_values.append(fps)
                print(f"FPS: {fps:.2f}, Average FPS: {np.mean(fps_values):.2f}")
                fps_start_time = time.time()
                frame_count = 0

        elif mode == "exit_battle":
            exit_battle()
            mode = None
            time.sleep(0.1)

        if recording and time.time() - time_since_last_recording >= 0.5:
            # Adjust the region to exclude the toolbar
            app_window_screenshot = controller.screenshot()

            # Convert the color channels from BGR to RGB
            cv2.imwrite(f"images/recording/{recording_date}/{record_number}.png", app_window_screenshot)
            record_number += 1
            time_since_last_recording = time.time()

        if live_detection:
            app_window_screenshot = controller.screenshot()

            found_troops = find_troops(app_window_screenshot)
            if True:
                # Plot the result on the screenshot
                im_array = found_troops[1].plot()

                # converts the image to an opencv image
                image = np.array(im_array)

                cv2.putText(image, f"Elixir: {partie.elixir_bleu}", (10, 790),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # shows name of every card in hand in top left corner, one line per card name
                for i in range(4):
                    cv2.putText(image, f"{partie.cartes_en_main[i]}", (10, (10 + i * 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # # put a blue transparent rectangle over the blue placement area, the rectangle is transparent and we cann se the rest underneath it
                # cv2.rectangle(image, partie.zone_placement_bleu[0][0], partie.zone_placement_bleu[0][1],
                #               (255, 0, 0, 25), -1)

                # in top right corner show in blue the number of blue crowns left and in red the number of red crowns left
                cv2.putText(image, f"Tours bleu: {partie.tours_bleu}", (770, 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, f"Tours rouge: {partie.tours_rouge}", (770, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                # Display the image using OpenCV
                cv2.imshow("YOLOv8 Inference", image)
                cv2.waitKey(1)


start()
# print("starting test")
# start_time = time.time()
# times = []
# for i in range(100):
#     print(i)
#     take_screenshot(rectangle_corners=((0, 0), (app_size[0], app_size[1] - (200))))
#     times.append(time.time() - start_time)
#     start_time = time.time()
# average = sum(times)/len(times)
#
# print("Average :", average)
