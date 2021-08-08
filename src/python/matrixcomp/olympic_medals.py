from enum import Enum
from itertools import Groupby

class OlympicMedal(Enum):
    Gold = 0
    Silver = 1
    Bronze = 2

    @staticmethod
    def sort_medals(medals: List["OlympicMedal"]) -> List["OlympicMedal"]:
        return sorted(medals, key = lambda medal: medal.value)

class OlympicMedals:

    def __init__(self, country: str, gold: int, silver: int, bronze:int) -> None:
        self.country = country
        self.gold = gold
        self.silver = silver
        self.bronze = bronze
        self.medal_count = gold + silver + bronze

    @staticmethod
    def from_medal_list(medals: List["OlympicMedal"]) -> OlympicMedals:
        sorted_list: List[OlympicMedal] = OlympicMedal.sort_medal(medals)
        grouped = groupby(sorted_list, key = lambda medal: medal.value)

