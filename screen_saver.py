import concurrent.futures
import datetime
import json
import os
import random
import re
import time

import cv2
import keyboard
import mss
import numpy as np
import pyautogui
import pygetwindow as gw
import pytesseract
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


def take_screenshot(resize=True, rectangle_corners=None):
    # Capture a screenshot
    # screenshot = pyautogui.screenshot( region=(app_window.left, app_window.top + toolbar_height, app_window.width, app_window.height-toolbar_height))
    if not rectangle_corners:
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
    return int(number * app_size[0] / 556)


def prop_height(number: int):
    return int(number * app_size[1] / 1019)


def get_elixir():
    # Define the color of empty squares and tolerance for matching
    empty_color = (5, 54, 123)
    color_tolerance = 20

    # Define the list of coordinates to check
    coordinates_to_check = [
        (prop_width(183) + app_coords[0], app_coords[1] + prop_height(998)),
        (prop_width(222) + app_coords[0], app_coords[1] + prop_height(998)),
        (prop_width(261) + app_coords[0], app_coords[1] + prop_height(998)),
        (prop_width(300) + app_coords[0], app_coords[1] + prop_height(998)),
        (prop_width(339) + app_coords[0], app_coords[1] + prop_height(998)),
        (prop_width(378) + app_coords[0], app_coords[1] + prop_height(998)),
        (prop_width(417) + app_coords[0], app_coords[1] + prop_height(998)),
        (prop_width(456) + app_coords[0], app_coords[1] + prop_height(998)),
        (prop_width(495) + app_coords[0], app_coords[1] + prop_height(998)),
        (prop_width(534) + app_coords[0], app_coords[1] + prop_height(998))
    ]

    # Function to check if a color matches the empty square color
    def is_empty_square(px_color):
        for i in range(3):
            if abs(px_color[i] - empty_color[i]) > color_tolerance:
                return False
        return True

    # Count the number of non-full squares
    non_full_count = 0

    for x, y in coordinates_to_check:
        pixel_color = pyautogui.pixel(x, y)
        if not is_empty_square(pixel_color):
            non_full_count += 1
    return non_full_count


def update_crowns():
    app_window_screenshot = take_screenshot(
        rectangle_corners=((prop_width(45), prop_height(180)), (prop_width(490), prop_height(750))))

    # Find the location of the template image in the screenshot
    location1 = find_image_in_screenshot(image_to_find1, app_window_screenshot)
    location2 = find_image_in_screenshot(image_to_find2, app_window_screenshot)

    if location1:
        return 2
    if location2:
        return 3
    # coords for the corners of the screenshots to take to get only the numbers
    numbers = [((prop_width(512), prop_height(534)), (prop_width(540), prop_height(564))),
               ((prop_width(513), prop_height(336)), (prop_width(540), prop_height(365)))]

    blue_number_image = take_screenshot(resize=False, rectangle_corners=numbers[0])
    red_number_image = take_screenshot(resize=False, rectangle_corners=numbers[1])
    # convert to gray scale
    blue_number_image = cv2.cvtColor(blue_number_image, cv2.COLOR_BGR2GRAY)
    red_number_image = cv2.cvtColor(red_number_image, cv2.COLOR_BGR2GRAY)
    found = False
    for i in range(2):
        image = numbers_images[i]
        # convert image to gray
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(blue_number_image, image, cv2.TM_CCOEFF_NORMED)
        threshold = 0.80  # Adjust this threshold as needed

        loc = np.where(result >= threshold)

        # click image if good enough
        if loc[0].any():
            if i == 0:
                crown_number = 1
            else:
                crown_number = 2
            if partie.tours_rouge > 3 - int(crown_number):
                partie.red_king_activated = True
                partie.tours_rouge = 3 - int(crown_number)
                found = True
                if partie.tours_rouge == 1:
                    partie.pv_tours_rouge[0] = None
                    partie.pv_tours_rouge[1] = None
                    partie.zone_placement_bleu = [((77, 351), (496, 750))]
                break

    for i in range(2, 4):
        image = numbers_images[i]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(red_number_image, image, cv2.TM_CCOEFF_NORMED)
        threshold = 0.80  # Adjust this threshold as needed

        loc = np.where(result >= threshold)

        # click image if good enough
        if loc[0].any():
            if i == 2:
                crown_number = 1
            else:
                crown_number = 2
            if partie.tours_bleu > 3 - int(crown_number):
                partie.blue_king_activated = True
                partie.tours_bleu = 3 - int(crown_number)
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
        print("Icone d'amis trouvé, démarrage du combat...")
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
            time.sleep(0.1)
        time.sleep(2.3)
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


def find_troops(screenshot, show=False):
    results: Results = model.predict(source=screenshot, stream=True, conf=0.6)
    for result in results:
        if show:
            # Plot the result on the screenshot
            im_array = result.plot()

            # converts the image to an opencv image
            image = np.array(im_array)

            # Display the image using OpenCV
            cv2.imshow("YOLOv8 Inference", image)
            cv2.waitKey(1)
        detections = []
        results_json = result.tojson()
        try:
            results_data = json.loads(results_json)
        except json.JSONDecodeError:
            print("Error decoding JSON:", results_json)
            return []

        for item in results_data:
            try:
                box = item['box']
            except IndexError:
                continue
            except TypeError as e:
                print(result.tojson())
                raise e
            center_x = (box['x1'] * opposite_scale_factor + box['x2'] * opposite_scale_factor) / 2
            center_y = (box['y1'] * opposite_scale_factor + box['y2'] * opposite_scale_factor) / 2
            # print(str(item['name']), box['x1'], box['x2'], (box['x1']* opposite_scale_factor + box['x2']* opposite_scale_factor) / 2)

            detections.append((item['name'], (center_x, center_y)))
        return detections


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

        text = pytesseract.image_to_string(thresholded, config='--psm 13')
        text = clean_string(text)

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
    coords = [((401, 169), (436, 186)), ((110, 169), (145, 186)), ((282, 46), (331, 69)),
              ((401, 636), (449, 654)), ((111, 637), (159, 653)), ((279, 777), (329, 796))]
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
        for card in all_cards:
            card_image = load_image(f"images/cards/{card}.png")
            if find_image_in_screenshot(card_image, card_slot):
                partie.cartes_en_main[i] = card
                break

    next_card_slot = take_screenshot(rectangle_corners=next_card_slot_box)
    for card in all_cards:
        card_image = load_image(f"images/cards/{card}.png")
        card_image = cv2.resize(card_image, (0, 0), fx=0.45, fy=0.45)
        if find_image_in_screenshot(card_image, next_card_slot):
            partie.prochaine_carte = card
            break


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
    if side=="left":
        return 3
    elif side=="right":
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

    print("Chosen side :", target_tower)

    print("Elixir :", partie.elixir_bleu)
    print("Current win conditions :", win_conditions, "Current support cards :", current_support_cards,
          "Current cycle cards :", current_cycle_cards, "Current building cards :", current_building_cards)
    if game_phase:  # pour l'instant, on ne vérifie pas la game phase, car ça ne sert à rien vu la complexité du bot
        plan = None
        attacking_enemy_troops = [troop for troop in enemy_troops["all"][0] if
                                  enemy_troops["all"][1][enemy_troops["all"][0].index(troop)][1] >
                                  BLUE_BRIDGE_COORDS["left"][1]]
        if attacking_enemy_troops:
            plan = "defend"
            print("Mode: Defense")
        else:
            plan = "attack"
            print("Mode: Attack")
            print("enemy troops : ",attacking_enemy_troops)

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
            if partie.elixir_bleu >= 8:
                if current_win_conditions and current_support_cards:
                    place_card(cards_in_hand.index(current_win_conditions[0]), BLUE_BRIDGE_COORDS[target_tower])
                    place_card(cards_in_hand.index(current_support_cards[0]),
                               (BLUE_BRIDGE_COORDS[target_tower][0], BLUE_BRIDGE_COORDS[target_tower][1] + 15))
                    print(f"Placing combo {current_win_conditions[0].split('ally_')[1]} and {current_support_cards[0].split('ally_')[1]} to push.")
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
                tower_distance = distance(partie.position_tours_bleu[blue_tower_index(get_side(troop_coord[0]))], troop_coord)
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
                if distance(partie.position_tours_bleu[blue_tower_index(get_side(pos_giant[0]))], pos_giant) > TILE_SIZE * 5 and "ally_goblin_cage" in cards_in_hand and partie.elixir_bleu >= 4 and ("enemy_giant" in enemy_troops["all"][0] or "enemy_prince" in enemy_troops["all"][0] or "enemy_mini_pekka" in enemy_troops["all"][0]):
                    place_card(cards_in_hand.index("ally_goblin_cage"), BLUE_MIDDLE_COORDS)
                    print(f"Défending with {pos_giant} with Goblin Cage")
                else:
                    if "ally_mini_pekka" in cards_in_hand and partie.elixir_bleu >= 4 and ("enemy_giant" in enemy_troops["all"][0] or "enemy_prince" in enemy_troops["all"][0] or "enemy_mini_pekka" in enemy_troops["all"][0]):
                        place_card(cards_in_hand.index("ally_mini_pekka"),
                                   (pos_giant[0], pos_giant[1] + int(TILE_SIZE * 2)))
                        print("Défending Giant with Mini Pekka")
                    elif "ally_goblin" in cards_in_hand and partie.elixir_bleu >= 2:
                        place_card(cards_in_hand.index("ally_goblin"),
                                   (pos_giant[0], pos_giant[1] + int(TILE_SIZE * 2)))
                        print("Défending Giant with Mini Pekka")
                    elif "ally_archer" in cards_in_hand and partie.elixir_bleu >= 3:
                        place_card(cards_in_hand.index("ally_archer"),
                                   (pos_giant[0], pos_giant[1] + int(TILE_SIZE * 2)))
                        print("Défending Giant with Mini Pekka")


app_window = gw.getWindowsWithTitle('LDPlayer')[0]
# shows app coords
print(f"Ready! (x={app_window.left}, y={app_window.top}) and (width={app_window.width}, height={app_window.height})")
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

image_to_find1 = load_image('images/battle/three_crown_blue.png')
image_to_find2 = load_image('images/battle/three_crown_red.png')
friends_icon = load_image('images/battle/friends_icon.png')
blason_de_combat = load_image('images/battle/vs_blason_debut_de_combat.png')
exit_battle_red_cross_button = load_image('images/battle/exit_battle_red_cross_button.png')
broken_tower = load_image('images/battle/broken_tower.png')

BLUE_MIDDLE_COORDS = (prop_width(265), prop_height(560))
BLUE_BRIDGE_COORDS = {"left": (prop_width(128), prop_height(438)), "right": (prop_width(419), prop_height(457))}
BLUE_BACK_COORDS = {"left": (prop_width(47), prop_height(750)), "right": (prop_width(502), prop_height(751))}


def start():
    mode = None
    timer = None
    recording = False
    record_number = 0
    live_detection = False

    while True:
        if mode != "battle" and keyboard.is_pressed('&'):  # 1
            exit()
        elif mode != "battle" and keyboard.is_pressed('é'):  # 2
            mode = "single_screenshot"
        elif mode != "battle" and keyboard.is_pressed('"') and app_window.isActive:  # 3
            mode = "start_battle"
        elif mode != "battle" and keyboard.is_pressed("'") and app_window.isActive:  # 4
            mode = "force_start_battle"
        elif mode != "battle" and keyboard.is_pressed("(") and app_window.isActive:  # 5
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
        elif mode != "battle" and keyboard.is_pressed("-") and app_window.isActive:  # 6
            if live_detection:
                live_detection = False
                print("stopped detecting")
                time.sleep(0.1)
            else:
                live_detection = True
                print("started detecting")
                time.sleep(0.1)

        if mode != "battle" and win32api.GetKeyState(0x02) < 0 and app_window.isActive:
            print("prop_width(", pyautogui.position().x - app_coords[0], ")", "prop_height(",
                  pyautogui.position().y - app_coords[1], ")")
            time.sleep(0.1)
        # check if middle click if yes, print current time minus timer
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

        if mode != "battle" and mode == "single_screenshot":
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
                coords = [((401, 169), (436, 186)), ((110, 169), (145, 186)), ((279, 777), (329, 796)),
                          ((282, 46), (331, 69)), ((401, 636), (449, 654)), ((111, 637), (159, 653))]
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
            print("Loop")
            all_troops = find_troops(
                take_screenshot(rectangle_corners=((0, 0), (app_size[0], app_size[1] - prop_height(200)))))
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
            partie.elixir_bleu = get_elixir()

            crowns_status = update_crowns()
            # print(partie.tours_bleu, partie.tours_rouge)
            if crowns_status == 2:
                print("partie finie par victoire bleu")
                mode = None
                continue
            if crowns_status == 3:
                print("partie finie par victoire rouge")
                mode = None
                continue

            if crowns_status:
                print("bleu", partie.tours_bleu, "rouge", partie.tours_rouge)
                if partie.tours_bleu == partie.tours_rouge:
                    print("partie finie par égalité")
                    mode = None
                    continue
                elif partie.tours_bleu > partie.tours_rouge:
                    print("partie finie par victoire bleu")
                    mode = None
                    continue
                elif partie.tours_bleu < partie.tours_rouge:
                    print("partie finie par victoire rouge")
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
                            print("partie finie par égalité")
                            mode = None
                            continue
                        elif partie.tours_bleu > partie.tours_rouge:
                            print("partie finie par victoire bleu")
                            mode = None
                            continue
                        elif partie.tours_bleu < partie.tours_rouge:
                            print("partie finie par victoire rouge")
                            mode = None
                            continue

                # print(partie.chrono)

            detect_tower_hp()

            get_current_cards()

            bot(troops, ally_troops)
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

            find_troops(app_window_screenshot, True)


start()
