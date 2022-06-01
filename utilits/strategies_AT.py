##################################################################################
# Copyright © 2021-2099 Ekosphere. All rights reserved
# File: strategies.py
# Author: Andrew Tsyrkunov
# Contacts: <andrey.pwn@gmail.com>
##################################################################################

"""
Модуль :mod:`~aitlf.strategies.strategies` содержит классы с реализацией нескольких базовых стратегий торговли.

Стратегии необходимы для узлов класса Evaluator для выполнения тестирования результатов разметки
или обучения нейронной сети с применением бэктестинга.

bt = Backtest(signals, Long_n_Short_Strategy, cash=1000000, commission=comm, trade_on_close=True)

"""

import math
from backtesting import Strategy
from loguru import logger
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG


class Long_n_Short_Strategy_Float(Strategy):
    """
      Простая стратегия для тестирования сети.
      'Длинные' и 'короткие' сделки.

      """

    fix_sum = 0
    deal_amount = "capital"
    signal_signature = {"entry_signal": 1, "exit_signal": -1}
    preview_signal = 1
    n1 = 10
    n2 = 20

    def __is_entry_signal(self, signal: int) -> bool:
        return self.signal_signature["entry_signal"] == signal

    def __is_exit_signal(self, signal: int) -> bool:
        return self.signal_signature["exit_signal"] == signal

    def __deal_size(self, price: float) -> float:
        """ Определяем размер сделки. """
        # Проверяем как выполняются сделки "На всю сумму" или "На фиксированную сумму".
        if self.deal_amount == "fix":
            # Проверяем хватает ли выделенной фиксированной суммы на покупку хотя бы одной акции
            if self.fix_sum // price:
                # Вычисляем размер сделки исходя из стоимости акций и имеющегося капитала
                deal_size = self.fix_sum // price if self.fix_sum <= self.equity else 0
            else:
                logger.warning(str(self.fix_sum) + " " + str(price))
                logger.warning(
                    "Фиксированная сумма покупки превышает размер капитала!!! Купить не могу!!!"
                )
                deal_size = 0
        else:
            deal_size = self.equity // price
        return deal_size

    def __sigmoid(self, x):
        return round(1 / (1 + math.exp(-x)), 0)

    def init(self):
        self.signal = self.I(
            lambda x: x, self.data.df.Signal, name="Signal", overlay=False
        )

    def next(self):
        # Цена закрытия предыдущая перед сигналом, на основании которого принимается решение о сделке
        price = self.data.Open[-1]
        # Крайний полученный сигнал
        last_signal = self.signal[-1]

        if self.__is_entry_signal(last_signal):
            if not self.position.is_long:
                self.position.close()
                self.buy(size=self.__deal_size(price))

        elif self.__is_exit_signal(last_signal):
            if not self.position.is_short:
                self.position.close()
                self.sell(size=self.__deal_size(price))
