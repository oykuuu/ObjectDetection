""" Takes a game image and champion images, finds the names
of the players in the two teams. Uses a computer vision approach.
"""

import cv2
import os
import json
import argparse
import numpy as np

from utils.utils import show

class LoLChampionDetector(object):
    """ Resize and pad champion images to increase training size.
    """

    def __init__(
            self,
            game_img_path="../screenshot.png",
            champion_folder="../champions/",
            n_players=5,
    ):
        """ Establishes game image and champion images.

        Parameters
        ----------
        game_img_path
            string, path to game image
        champion_folder
            string, path to champions folder

        Returns
        -------
        None
        """

        self.game = cv2.imread(game_img_path, cv2.IMREAD_COLOR)
        self.champion_paths = [
            os.path.join(champion_folder, ch_file)
            for ch_file in os.listdir(champion_folder)
        ]
        self.champion_names = [
            ch_file.split(".")[0] for ch_file in os.listdir(champion_folder)
        ]
        self.n_champions = len(self.champion_names)
        self.n_players = n_players

    def split_game_image(self, adjustment=5, width=50):
        """
        Find the leaderboard within the screenshot and
        split it in half to isolate two teams.

        Parameters
        ----------
        adjustment
            int, adjustment pixels as image leans towards right
        width
            int, width (in pixels) of the extracted region

        Returns
        -------
        team_left
            np.array, left half of the screenshot
        team_right
            np.array, right half of the screenshot
        """
        mid = self.game.shape[1] // 2 + adjustment
        team_left = self.game[850:, mid - width : mid, :].copy()
        team_right = self.game[850:, mid : mid + width + 1, :].copy()

        return team_left, team_right

    def match_champion(self, champion_file, team, scaled_size):
        """
        Since template size and the object sizes in game screenshot
        are different, resizes the template (champion). Then carries
        out template matching.

        Parameters
        ----------
        champion_file
            string, filepath of the champion
        team
            np.array, image of the left or the right team
        scaled_size
            int, new scaled size of the template (champion)

        Returns
        -------
        max_val
            float, normalized matching score in range [0,1] with 1 being
            the perfect match.
        """
        champion = cv2.imread(champion_file, cv2.IMREAD_COLOR)
        scaled_champion = cv2.resize(champion, (scaled_size, scaled_size))
        match = cv2.matchTemplate(team, scaled_champion, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, top_left = cv2.minMaxLoc(match)

        return max_val

    def compare_all_champions(self, team, all_sizes):
        """
        Try template matching of all champions at different scalings of
        the champions. Store all the matched prediction scores in a dictionary
        where a key represents the new scaled size of champions and the value
        is an array containing all the scores for champion matchings at new size.

        Parameters
        ----------
        team
            np.array, image of the left or the right team
        all_sizes
            list, list of scaling factors for champion images

        Returns
        -------
        predictions
            dict, dictionary of match scores at different scales
        """
        predictions = {}
        for new_size in all_sizes:
            key_name = "size_" + str(new_size)
            pred_fixed_size = []
            for champion_file in self.champion_paths:
                max_val = self.match_champion(champion_file, team, new_size)
                pred_fixed_size += [max_val]

            predictions[key_name] = [self.champion_names, pred_fixed_size]

        return predictions

    def find_best_scale(self, predictions):
        """
        Given all the predictions for champions scaled at different sizes,
        detects the best size that results in the best average scores for
        the top n players.

        Parameters
        ----------
        team
            np.array, image of the left or the right team

        Returns
        -------
        best_size
            int, size of the template that gave the best overall match scores.
        """
        best_size = None
        best_size_val = -1

        for key_name, _ in predictions.items():
            top_match_index = np.array(predictions[key_name][1]).argsort()[-self.n_champions:][::-1]
            top_match_champions = [
                predictions[key_name][0][id] for id in top_match_index
            ]
            top_match_vals = [predictions[key_name][1][id] for id in top_match_index]

            if np.mean(top_match_vals[: self.n_players]) > best_size_val:
                best_size_val = np.mean(top_match_vals[: self.n_players])
                best_size = int(key_name.split("_")[-1])

        return best_size

    def get_matched_players(self, predictions, size):
        """
        For a given scaled template, provides the names of the top matched players.

        Parameters
        ----------
        team
            np.array, image of the left or the right team

        Returns
        -------
        best_size
            int, size of the template that gave the best overall match scores.
        """
        key_name = "size_" + str(size)
        top_match_index = np.array(predictions[key_name][1]).argsort()[
            -self.n_players :
        ][::-1]
        top_match_champion_names = [
            predictions[key_name][0][id] for id in top_match_index
        ]
        top_match_vals = [predictions[key_name][1][id] for id in top_match_index]
        predicted_players = top_match_champion_names[: self.n_players]
        return predicted_players, top_match_champion_names, top_match_vals


def main(config_path):
    config = json.load(open(config_path, "r"))
    game_img_path = config["paths"]["game_image_path"]
    champion_folder = config["paths"]["champion_folder"]
    min_template_size = config["cv"]["min_template_size"]
    max_template_size = config["cv"]["max_template_size"]

    all_sizes = np.arange(min_template_size, max_template_size + 1)

    lol_object = LoLChampionDetector(game_img_path, champion_folder)
    team_left, team_right = lol_object.split_game_image()
    left_predictions = lol_object.compare_all_champions(team_left, all_sizes)
    right_predictions = lol_object.compare_all_champions(team_right, all_sizes)

    best_size = lol_object.find_best_scale(left_predictions)
    print("Detected best size to scale champions as {} pixels.\n".format(best_size))

    left_players, ordered_champions, ordered_vals = lol_object.get_matched_players(
        left_predictions, best_size
    )
    right_players, ordered_champions, ordered_vals = lol_object.get_matched_players(
        right_predictions, best_size
    )
    print("Left team members are predicted as: \n {}\n".format(left_players))
    print("Right team members are predicted as: \n {}\n".format(right_players))


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "-c", "--config", help="filepath to config json", default="./config.json"
    )
    ARGS = PARSER.parse_args()
    CONFIGPATH = ARGS.config
    main(CONFIGPATH)
