##################################################################################
# Copyright © 2021-2099 Ekosphere. All rights reserved
# Author: Evgeny Matusevich
# Contacts: <ma2sevich222@gmail.com>
##################################################################################

import pandas as pd
from other_codes.not_used.visuals_functions import plot_prediction

pd.pandas.set_option("display.max_columns", None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)

source_root = "outputs"
test_file_name = "test_results_extrw90_patsize30.csv"

sell_trash = 1.6
hold_trash_per = [0.5, 1.6]
buy_trash = 0.5

try:

    df = pd.read_csv(f"{source_root}/{test_file_name}")
    plot_prediction(df, sell_trash, hold_trash_per, buy_trash, test_file_name)


except FileNotFoundError:
    print("Файл для анализа предсказаний с учетом профита отсутсвует")
