import concurrent.futures
import datetime
import json
import os
import random
import re
import time
import dxcam
import cv2
import easyocr
import keyboard
import mss
import numpy as np
import pyautogui
import pygetwindow as gw
import win32api
from ultralytics import YOLO
from ultralytics.engine.results import Results


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


def find_image_in_screenshot(template, screenshot, return_coords=False, threshold=0.80):
    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)

    if loc[0].any():
        if return_coords:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            h, w = template.shape[:-1]
            w, h = int(w * opposite_scale_factor), int(h * opposite_scale_factor)

            center_x = int(max_loc[0] * opposite_scale_factor + app_window.left + w / 2)
            center_y = int(max_loc[1] * opposite_scale_factor + app_window.top + h / 2)
            return center_x, center_y
        return True
    return False


def get_side(x):
    middle = int(app_size[0] / 2)
    if x < middle:
        return "left"
    else:
        return "right"


def distance(coords1, coords2):
    x1, y1 = coords1
    x2, y2 = coords2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def backup_screenshot(resize=True, rectangle_corners=None, full_screen=False):
    if full_screen:
        monitor = (0, 0, screen_width, screen_height)
    # Capture a screenshot
    elif not rectangle_corners:
        monitor = {"top": app_window.top + toolbar_height, "left": app_window.left, "width": app_window.width,
                   "height": app_window.height - toolbar_height}

    else:
        monitor = {"top": app_coords[1] + rectangle_corners[0][1], "left": app_coords[0] + rectangle_corners[0][0],
                   "width": rectangle_corners[1][0] - rectangle_corners[0][0],
                   "height": rectangle_corners[1][1] - rectangle_corners[0][1]}

    with mss.mss() as sct:
        screenshot = np.array(sct.grab(monitor))[:, :, :3]

    if resize:
        screenshot = cv2.resize(np.array(screenshot), (0, 0), fx=scale_factor, fy=scale_factor)
    return screenshot


def take_screenshot(resize=True, rectangle_corners=None, full_screen=False):
    if full_screen:
        monitor = (0, 0, screen_width, screen_height)
    elif not rectangle_corners:
        left = app_window.left
        top = app_window.top + toolbar_height
        right = left + app_window.width
        bottom = top + (app_window.height - toolbar_height)
        monitor = (left, top, right, bottom)
    else:
        left = app_coords[0] + rectangle_corners[0][0]
        top = app_coords[1] + rectangle_corners[0][1]
        width = rectangle_corners[1][0] - rectangle_corners[0][0]
        height = rectangle_corners[1][1] - rectangle_corners[0][1]
        monitor = (left, top, left + width, top + height)
    screenshot = camera.grab(region=monitor)
    # Convert to numpy array and RGB format
    screenshot = np.array(screenshot)
    if screenshot.dtype == "object":
        return backup_screenshot(resize, rectangle_corners, full_screen)

    if resize:
        screenshot = cv2.resize(screenshot, (0, 0), fx=scale_factor, fy=scale_factor)

    return screenshot


def load_image(image_path, resize=True):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.resize(np.array(image), (prop_width(image.shape[1]), prop_height(image.shape[0])))
    if resize:
        image = cv2.resize(np.array(image), (0, 0), fx=scale_factor, fy=scale_factor)

    return image


def current_time():
    time_now = datetime.datetime.now()
    # Convert the date into a valid file name
    time_now = time_now.strftime("%Y-%m-%d %H-%M-%S")

    return time_now


def prop_width(number: int):
    return int(number * (app_size[0] / 556))


def prop_height(number: int):
    return int(number * (app_size[1] / 1019))


from PIL import ImageGrab
import numpy as np

import numpy as np
import cv2

def get_elixir():
    # Define the color of empty squares and tolerance for matching
    empty_color = np.array([123, 54, 5])
    color_tolerance = 20

    # Precompute app_coords offset
    x_offset = app_coords[0]
    y_base = app_coords[1] + prop_height(993)

    # Define the list of x-coordinates to check
    x_coordinates = [183, 222, 261, 300, 339, 378, 417, 456, 495, 534]
    coordinates_to_check = [(prop_width(x) + x_offset, y_base) for x in x_coordinates]

    # Capture the screen once
    screen = take_screenshot(resize=False, full_screen=True)

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
    app_window_screenshot = take_screenshot(
        rectangle_corners=((prop_width(45), prop_height(180)), (prop_width(490), prop_height(750))))

    # Find the location of the template image in the screenshot
    location1 = find_image_in_screenshot(blue_three_crown_image, app_window_screenshot)
    location2 = find_image_in_screenshot(red_three_crown_image, app_window_screenshot)

    if location1:
        return 2
    if location2:
        return 3

    # Coordinates for the corners of the screenshots to get only the numbers
    numbers = [((prop_width(512), prop_height(534)), (prop_width(540), prop_height(564))),
               ((prop_width(513), prop_height(336)), (prop_width(540), prop_height(365)))]

    blue_number_image = cv2.cvtColor(take_screenshot(resize=False, rectangle_corners=numbers[0]), cv2.COLOR_BGR2GRAY)
    red_number_image = cv2.cvtColor(take_screenshot(resize=False, rectangle_corners=numbers[1]), cv2.COLOR_BGR2GRAY)

    found = False
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in numbers_images]
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
            box_image = take_screenshot(rectangle_corners=box)
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
            box_image = take_screenshot(rectangle_corners=box)
            coords = find_image_in_screenshot(broken_tower, box_image, return_coords=True)
            if coords:
                side = get_side(coords[0])
                if side == "left":
                    partie.pv_tours_rouge[0] = None
                    partie.zone_placement_bleu = [((77, 480), (496, 750)), [(77, 480), (283, 750)]]
                else:
                    partie.pv_tours_rouge[1] = None
                    partie.zone_placement_bleu = [((77, 480), (496, 750)), [(293, 480), (496, 750)]]

    return found and partie.overtime



def start_battle():
    app_window_screenshot = take_screenshot(
        rectangle_corners=((prop_width(323), prop_height(102)), (prop_width(401), prop_height(173))))
    # Find the location of the template image in the screenshot
    location = find_image_in_screenshot(friends_icon, app_window_screenshot)

    if location:
        print("Friends icon found, starting battle...")
        time.sleep(0.1)
        pyautogui.click(app_coords[0] + prop_width(519), app_coords[1] + prop_height(139))
        time.sleep(0.1)
        pyautogui.click(app_coords[0] + prop_width(348), app_coords[1] + prop_height(349))
        time.sleep(0.1)
        pyautogui.click(app_coords[0] + prop_width(370), app_coords[1] + prop_height(599))

        app_window_screenshot = take_screenshot(
            rectangle_corners=((prop_width(175), prop_height(375)), (prop_width(375), prop_height(550))))

        # Find the location of the template image in the screenshot
        location = find_image_in_screenshot(blason_de_combat, app_window_screenshot)
        while not location:
            app_window_screenshot = take_screenshot(
                rectangle_corners=((prop_width(175), prop_height(375)), (prop_width(375), prop_height(550))))

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
        return False


def exit_battle():
    app_window_screenshot = take_screenshot(
        rectangle_corners=((prop_width(16), prop_height(827)), (prop_width(88), prop_height(904))))

    # Find the location of the template image in the screenshot
    location = find_image_in_screenshot(exit_battle_red_cross_button, app_window_screenshot)

    if location:
        print("Exiting battle...")
        time.sleep(0.1)
        pyautogui.click(app_coords[0] + prop_width(54), app_coords[1] + prop_height(866))
        time.sleep(0.1)
        pyautogui.click(app_coords[0] + prop_width(384), app_coords[1] + prop_height(636))
        time.sleep(7)
        pyautogui.click(app_coords[0] + prop_width(270), app_coords[1] + prop_height(870))
        image_to_find = friends_icon
        app_window_screenshot = take_screenshot(
            rectangle_corners=((prop_width(323), prop_height(102)), (prop_width(401), prop_height(173))))

        # Find the location of the template image in the screenshot
        location = find_image_in_screenshot(image_to_find, app_window_screenshot)
        while not location:
            app_window_screenshot = take_screenshot(
                rectangle_corners=((prop_width(323), prop_height(102)), (prop_width(401), prop_height(173))))

            # Find the location of the template image in the screenshot
            location = find_image_in_screenshot(image_to_find, app_window_screenshot)


import json

def find_troops(screenshot):
    results: Results = model.predict(source=screenshot, stream=True, conf=0.4, verbose=False)
    for result in results:
        detections = []
        results_json = result.tojson()
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
                center_x = (box['x1'] + box['x2']) * opposite_scale_factor / 2
                center_y = (box['y1'] + box['y2']) * opposite_scale_factor / 2
            except (KeyError, TypeError) as e:
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

        img = take_screenshot(resize=False, rectangle_corners=coord)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresholded = cv2.bitwise_not(gray)

        numbers = []
        text = reader.readtext(thresholded)[0][1]

        #text = clean_string(text)

        if text == '' or text == " ":
            return None, None

        hp = int(text)
        if i < 3:
            if partie.pv_tours_rouge[i] and hp < partie.pv_tours_rouge[i] and partie.pv_tours_rouge[i] - hp < 1400:
                return i, hp
        else:
            if partie.pv_tours_bleu[i] and hp < partie.pv_tours_bleu[i] and partie.pv_tours_bleu[i] - hp < 1400:
                return i, hp
    except:
        pass
    return None, None


def filter_not_none(item):
    i, text = item
    if i is not None and text is not None:
        return True
    return False


def detect_tower_hp():
    coords = [((401, 166), (436, 190)), ((110, 166), (145, 190)), ((282, 46), (331, 69)),
              ((401, 640), (449, 660)), ((111, 640), (159, 660)), ((279, 777), (329, 796))]
    coords = [((prop_width(x1), prop_height(y1)), (prop_width(x2), prop_height(y2))) for (x1, y1), (x2, y2) in coords]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        parse_results = list(executor.map(parse_image, range(len(coords)), coords))

    for i, text in filter(filter_not_none, parse_results):
        if i < 3:
            if text:
                partie.pv_tours_rouge[i] = text
        else:
            if text:
                partie.pv_tours_bleu[i] = text


def place_card(card_slot_index, pos):
    pyautogui.click(app_coords[0] + card_slots_positions[card_slot_index][0],
                    app_coords[1] + card_slots_positions[card_slot_index][1])
    time.sleep(0.01)
    pyautogui.click(app_coords[0] + pos[0], app_coords[1] + pos[1])
    partie.elixir_bleu -= all_cards_cost[partie.cartes_en_main[card_slot_index]]
    partie.cartes_en_main[card_slot_index] = None


def get_current_cards():
    for i in range(4):
        card_slot = take_screenshot(rectangle_corners=card_slot_boxes[i])
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


def bot(enemy_troops, ally_troops):
    # deck : Prince, Mini Pekka, Archers, Spear Goblins, Giant, Goblin Hut, Arrows
    cards_in_hand = list(partie.cartes_en_main.values())
    # print("Cards in hand :", cards_in_hand, "Elixir :", partie.elixir_bleu)
    current_win_conditions = [element for element in cards_in_hand if element in win_conditions]
    current_cycle_cards = [element for element in cards_in_hand if element in cycle_cards]
    current_building_cards = [element for element in cards_in_hand if element in buildings]
    current_support_cards = [element for element in cards_in_hand if element in support_cards]

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

    #print("Chosen side :", target_tower)

    #print("Elixir :", partie.elixir_bleu)
    #print("Current win conditions :", win_conditions, "Current support cards :", current_support_cards, "Current cycle cards :", current_cycle_cards, "Current building cards :", current_building_cards)
    if game_phase:  # pour l'instant, on ne vérifie pas la game phase, car ça ne sert à rien vu la complexité du bot
        plan = None
        attacking_enemy_troops = [troop for troop in enemy_troops["all"][0] if
                                  enemy_troops["all"][1][enemy_troops["all"][0].index(troop)][1] >
                                  BLUE_BRIDGE_COORDS["left"][1]]
        if attacking_enemy_troops:
            plan = "defend"
            # print("Mode: Defense")
        else:
            plan = "attack"
            # print("Mode: Attack")
            if attacking_enemy_troops:
                print("enemy troops : ", attacking_enemy_troops)

        # do for any plan
        if plan:
            if "ally_arrow" in cards_in_hand and partie.elixir_bleu >= all_cards_cost["ally_arrow"] and len(
                    enemy_troops["all"][0]) >= 3:
                optimal_center, count = find_optimal_center(enemy_troops["all"][1], int(2 * TILE_SIZE))
                if count >= 3:
                    place_card(cards_in_hand.index("ally_arrow"),
                               (optimal_center[0], optimal_center[1] + int(TILE_SIZE * 3.5)))
                    print(f"Found cluster of {count} troops in {optimal_center}, playing arrows.")

        # do for attack plan
        if plan == "attack":
            if partie.elixir_bleu >= 9:
                if current_win_conditions and current_support_cards:
                    place_card(cards_in_hand.index(current_win_conditions[0]), BLUE_BRIDGE_COORDS[target_tower])
                    place_card(cards_in_hand.index(current_support_cards[0]),
                               (BLUE_BRIDGE_COORDS[target_tower][0], BLUE_BRIDGE_COORDS[target_tower][1] + 15))
                    print(
                        f"Placing combo {current_win_conditions[0].split('ally_')[1]} and {current_support_cards[0].split('ally_')[1]} to push.")
                elif current_cycle_cards:
                    place_card(cards_in_hand.index(current_cycle_cards[0]), BLUE_BACK_COORDS[target_tower])
                    print(f"Placing {current_cycle_cards[0].split('ally_')[1]} to cycle.")


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

            print(troop_to_defend, "is attacking")
            pos_giant = closest_troop_coord
            no_one_already_defending = True
            try:
                goblin_cage_defending = "ally_gobelin_cage" in ally_troops[get_side(pos_giant[0])][0]
                mini_pekka_defending = "ally_mini_pekka" in ally_troops[get_side(pos_giant[0])][0]
            except Exception as e:
                print(ally_troops)
                print(ally_troops[get_side(pos_giant[0])])
                print(pos_giant[0])
                raise e
            if mini_pekka_defending:
                mini_pekka_defending_pos = ally_troops[get_side(pos_giant[0])][1][
                    ally_troops[get_side(pos_giant[0])][0].index("ally_mini_pekka")]
                if distance(mini_pekka_defending_pos, pos_giant) < TILE_SIZE * 6:
                    print("mini pekka already defending")
                    no_one_already_defending = False
            if goblin_cage_defending:
                print("goblin cage already defending")
                no_one_already_defending = False

            if no_one_already_defending:
                if distance(partie.position_tours_bleu[blue_tower_index(get_side(pos_giant[0]))],
                            pos_giant) > TILE_SIZE * 6 and "ally_goblin_cage" in cards_in_hand and partie.elixir_bleu >= 4 and (
                        "enemy_giant" in enemy_troops["all"][0] or "enemy_prince" in enemy_troops["all"][
                    0] or "enemy_mini_pekka" in enemy_troops["all"][0] or "enemy_knight" in enemy_troops["all"][0]):
                    place_card(cards_in_hand.index("ally_goblin_cage"), BLUE_MIDDLE_COORDS)
                    print(f"Defending with {pos_giant} with Goblin Cage")
                else:
                    if "ally_mini_pekka" in cards_in_hand and partie.elixir_bleu >= 4 and (
                            "enemy_giant" in enemy_troops["all"][0] or "enemy_prince" in enemy_troops["all"][
                        0] or "enemy_mini_pekka" in enemy_troops["all"][0]):
                        place_card(cards_in_hand.index("ally_mini_pekka"),
                                   (pos_giant[0], pos_giant[1] + int(TILE_SIZE * 2)))
                        print("Defending Giant with Mini Pekka")
                    elif "ally_goblin" in cards_in_hand and partie.elixir_bleu >= 2:
                        place_card(cards_in_hand.index("ally_goblin"),
                                   (pos_giant[0], pos_giant[1] + int(TILE_SIZE * 2)))
                        print("Defending Giant with Mini Pekka")
                    elif "ally_archer" in cards_in_hand and partie.elixir_bleu >= 3:
                        place_card(cards_in_hand.index("ally_archer"),
                                   (pos_giant[0], pos_giant[1] + int(TILE_SIZE * 2)))
                        print("Defending Giant with Mini Pekka")


app_window = gw.getWindowsWithTitle('LDPlayer')[0]
try:
    if app_window.isMinimized:
        app_window.restore()
    app_window.activate()
except:
    pass
app_coords = (app_window.left, app_window.top)
app_size = (app_window.width, app_window.height)
scale_factor = 1
opposite_scale_factor = 1 / scale_factor
# Define the toolbar height
toolbar_height = prop_height(34)
TILE_SIZE = prop_width(25)
# Load the image you want to find
model = YOLO("best (1).pt")  # load a pretrained model (recommended for training)
partie = Partie()

numbers_images = [cv2.imread("images/battle/un_bleu.png"), cv2.imread("images/battle/deux_bleu.png"),
                  cv2.imread("images/battle/un_rouge.png"), cv2.imread("images/battle/deux_rouge.png")]

card_slots_positions = [(prop_width(177), prop_height(913)), (prop_width(277), prop_height(913)),
                        (prop_width(377), prop_height(913)), (prop_width(477), prop_height(913))]

card_slot_boxes = [((prop_width(125), prop_height(848)), (prop_width(221), prop_height(971))),
                   ((prop_width(229), prop_height(848)), (prop_width(325), prop_height(971))),
                   ((prop_width(333), prop_height(848)), (prop_width(429), prop_height(971))),
                   ((prop_width(437), prop_height(848)), (prop_width(533), prop_height(971)))]

next_card_slot_box = ((prop_width(25), prop_height(948)), (prop_width(74), prop_height(1008)))
all_cards = ["ally_knight", "ally_archer", "ally_prince", "ally_goblin", "ally_spear_goblin", "ally_giant",
             "ally_mini_pekka", "ally_minion", "ally_arrow", "ally_fireball", "ally_goblin_cage", "ally_musketeer",
             "ally_goblin_hut"]
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
win_conditions = ["ally_prince", "ally_giant", "ally_mini_pekka"]
cycle_cards = ["ally_archer", "ally_spear_goblin", "ally_goblin", "ally_knight"]
buildings = ["ally_goblin_cage", "ally_goblin_hut"]
support_cards = ["ally_archer", "ally_minion", "ally_spear_goblin", "ally_goblin"]

blue_three_crown_image = load_image('images/battle/three_crown_blue.png')
red_three_crown_image = load_image('images/battle/three_crown_red.png')
friends_icon = load_image('images/battle/friends_icon.png')
blason_de_combat = load_image('images/battle/vs_blason_debut_de_combat.png')
exit_battle_red_cross_button = load_image('images/battle/exit_battle_red_cross_button.png')
broken_tower = load_image('images/battle/broken_tower.png')

BLUE_MIDDLE_COORDS = (prop_width(265), prop_height(560))
BLUE_BRIDGE_COORDS = {"left": (prop_width(128), prop_height(438)), "right": (prop_width(419), prop_height(457))}
BLUE_BACK_COORDS = {"left": (prop_width(47), prop_height(750)), "right": (prop_width(502), prop_height(751))}

preloaded_card_icons = {card: load_image(f"images/cards/{card}.png") for card in all_cards}
camera = dxcam.create(output_color="BGR")
reader = easyocr.Reader(['en'])
screen_width, screen_height = pyautogui.size()
print(f"Ready! (x={app_window.left}, y={app_window.top}) and (width={app_window.width}, height={app_window.height})")

def start():
    mode = None
    timer = None
    recording = False
    record_number = 0
    live_detection = False
    start_time = None
    fps_start_time = time.time()
    frame_count = 0
    while True:
        #1 exit
        if keyboard.is_pressed('&'):  # 1
            exit()

        #2 screenshot
        elif keyboard.is_pressed('é'):  # 2
            ss_mode = "multiple"
            if ss_mode == "single":
                current_date = current_time()

                start_time = time.time()

                # Adjust the region to exclude the toolbar
                app_window_screenshot = take_screenshot()
                # Convert the color channels from BGR to RGB
                cv2.imwrite(f"images/single_ss/{current_date}.png", app_window_screenshot)

                print(f"Screenshot taken and saved in {time.time() - start_time} seconds")
                print()
                mode = None
                time.sleep(0.5)
            if ss_mode == "multiple":
                coords = [((401, 162), (450, 185)), ((111, 162), (155, 185)), ((282, 46), (331, 69)),
                          ((401, 644), (450, 662)), ((111, 644), (155, 662)), ((279, 777), (329, 796))]
                coords = [((prop_width(x1), prop_height(y1)), (prop_width(x2), prop_height(y2))) for (x1, y1), (x2, y2)
                          in coords]

                for coord in coords:
                    current_date = current_time()

                    start_time = time.time()

                    # Adjust the region to exclude the toolbar
                    app_window_screenshot = take_screenshot(resize=False, rectangle_corners=coord)
                    # Convert the color channels from BGR to RGB
                    cv2.imwrite(f"images/single_ss/{current_date}_{coord[0][0]}.png", app_window_screenshot)

                    print(f"Screenshot taken and saved in {time.time() - start_time} seconds")
                    print()
                    mode = None
                    time.sleep(0.1)

        #3 Start battle
        elif mode != "battle" and keyboard.is_pressed('"') and app_window.isActive:  # 3
            mode = "start_battle"
        #4 Force Start battle
        elif mode != "battle" and keyboard.is_pressed("'") and app_window.isActive:  # 4
            mode = "force_start_battle"

        #5 toggle recording
        elif keyboard.is_pressed("(") and app_window.isActive:  # 5
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

        #6 toggle live detection
        elif keyboard.is_pressed("-") and app_window.isActive:
            if live_detection:
                live_detection = False
                print("stopped detecting")
                time.sleep(0.1)
            else:
                live_detection = True
                print("started detecting")
                time.sleep(0.1)


        if mode != "battle" and win32api.GetKeyState(0x02) < 0 :
            print("prop_width(", pyautogui.position().x - app_coords[0], ")", "prop_height(",
                  pyautogui.position().y - app_coords[1], ")")
            time.sleep(0.1)

        if mode != "battle" and win32api.GetKeyState(0x04) < 0 and app_window.isActive:
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
            found_troops = find_troops(
                take_screenshot(rectangle_corners=((0, 0), (app_size[0], app_size[1] - prop_height(200)))))
            find_troops_time = time.time() - start

            all_troops = found_troops[0]
            troops = {"left": ([], []), "right": ([], []), "all": ([], [])}
            ally_troops = {"left": ([], []), "right": ([], []), "all": ([], [])}
            for troop in all_troops:
                if troop[0].startswith("enemy_"):
                    if troop[1][1] <= int(app_size[1] / 2):
                        troops["left"][0].append(troop[0])
                        troops["left"][1].append(troop[1])
                    else:
                        troops["right"][0].append(troop[0])
                        troops["right"][1].append(troop[1])
                    troops["all"][0].append(troop[0])
                    troops["all"][1].append(troop[1])
                if troop[0].startswith("ally_"):
                    if troop[1][1] <= int(app_size[1] / 2):
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
            get_current_cards()
            get_current_cards_time = time.time() - start

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
                print(f"FPS: {fps:.2f}")
                fps_start_time = time.time()
                frame_count = 0

        elif mode == "exit_battle":
            exit_battle()
            mode = None
            time.sleep(0.1)

        if recording:
            # Adjust the region to exclude the toolbar
            app_window_screenshot = take_screenshot(
                rectangle_corners=((0, 0), (app_size[0], app_size[1] - prop_height(200))))

            # Convert the color channels from BGR to RGB
            cv2.imwrite(f"images/recording/{recording_date}/{record_number}.png", app_window_screenshot)
            record_number += 1
            time.sleep(0.5)

        if live_detection:
            app_window_screenshot = take_screenshot(
                rectangle_corners=((0, 0), (app_size[0], app_size[1] - prop_height(200))))

            found_troops = find_troops(app_window_screenshot)
            if True:
                # Plot the result on the screenshot
                im_array = found_troops[1].plot()

                # converts the image to an opencv image
                image = np.array(im_array)

                cv2.putText(image, f"Elixir: {partie.elixir_bleu}", (prop_width(10), prop_height(790)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # shows name of every card in hand in top left corner, one line per card name
                for i in range(4):
                    cv2.putText(image, f"{partie.cartes_en_main[i]}", (prop_width(10), prop_height(10 + i * 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # # put a blue transparent rectangle over the blue placement area, the rectangle is transparent and we cann se the rest underneath it
                # cv2.rectangle(image, partie.zone_placement_bleu[0][0], partie.zone_placement_bleu[0][1],
                #               (255, 0, 0, 25), -1)

                # in top right corner show in blue the number of blue crowns left and in red the number of red crowns left
                cv2.putText(image, f"Tours bleu: {partie.tours_bleu}", (prop_width(770), prop_height(10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, f"Tours rouge: {partie.tours_rouge}", (prop_width(770), prop_height(30)),
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
#     take_screenshot(rectangle_corners=((0, 0), (app_size[0], app_size[1] - prop_height(200))))
#     times.append(time.time() - start_time)
#     start_time = time.time()
# average = sum(times)/len(times)
#
# print("Average :", average)
