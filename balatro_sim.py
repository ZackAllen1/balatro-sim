import random
import math
from collections import Counter, OrderedDict
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple
import simpy
from helper import calculate_score_confidence_interval, print_statistical_results, save_to_excel

class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"

class Rank(Enum):
    TWO = (2, "2", 2)
    THREE = (3, "3", 3)
    FOUR = (4, "4", 4)
    FIVE = (5, "5", 5)
    SIX = (6, "6", 6)
    SEVEN = (7, "7", 7)
    EIGHT = (8, "8", 8)
    NINE = (9, "9", 9)
    TEN = (10, "10", 10)
    JACK = (10, "J", 11)
    QUEEN = (10, "Q", 12)
    KING = (10, "K", 13)
    ACE = (11, "A", 14)
    
    def __init__(self, value, display, order):
        self._value_ = value
        self.display = display
        self.rank_order = order # needed this to do fix issues with straightgs

@dataclass
class Card:
    rank: Rank
    suit: Suit
    
    def __str__(self):
        return f"{self.rank.display}{self.suit.value}"

class HandType(Enum):
    HIGH_CARD = (5, 1, "High Card")
    PAIR = (10, 2, "Pair")
    TWO_PAIR = (20, 2, "Two Pair")
    THREE_KIND = (30, 3, "Three of a Kind")
    STRAIGHT = (30, 4, "Straight")
    FLUSH = (35, 4, "Flush")
    FULL_HOUSE = (40, 4, "Full House")
    FOUR_KIND = (60, 7, "Four of a Kind")
    STRAIGHT_FLUSH = (100, 8, "Straight Flush")
    ROYAL_FLUSH = (100, 8, "Royal Flush")
    
    def __init__(self, chips, mult, name):
        self.chips = chips
        self.mult = mult
        self.handName = name

@dataclass
class BlindConfig:
    name: str
    target_score: int

@dataclass
class GameConfig:
    config_name: str
    hands_per_blind: int
    discards_per_blind: int
    antes: List[List[BlindConfig]]

@dataclass
class ActionRecord:
    game_number: int
    strategy_name: str
    config_name: str
    ante: int
    blind_name: str
    current_deck_size: int
    current_hand: str
    action: str
    cards_played: str
    hand_type: str
    score_for_play: int
    total_score_for_blind: int
    hands_remaining: int
    discards_remaining: int
    blind_defeated: str

class PokerEvaluator:
    @staticmethod
    def evaluate_hand(cards: List[Card]) -> Tuple[HandType, int]:
        """Evaluate poker hand and return hand type and chip value"""
        if len(cards) != 5:
            return HandType.HIGH_CARD, sum(card.rank.value for card in cards)
        
        ranks = [card.rank.rank_order for card in cards]
        suits = [card.suit for card in cards]
        values = [card.rank._value_ for card in cards]

        # count rank/suit frequencies
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        unique_ranks = sorted(set(ranks))
        is_flush = len(set(suit_counts)) == 1
        is_straight = len(unique_ranks) == 5 and max(unique_ranks) - min(unique_ranks) == 4
        
        chip_value = sum(values)
        
        # check for straight
        if set(ranks) == {2, 3, 4, 5, 14}:  # Ace-low straight (A,2,3,4,5)
            is_straight = True
        
        # determine hand type
        if is_flush and set(ranks) == {10, 11, 12, 13, 14}:
            return HandType.ROYAL_FLUSH, chip_value
        elif is_straight and is_flush:
            return HandType.STRAIGHT_FLUSH, chip_value
        elif 4 in rank_counts.values():
            return HandType.FOUR_KIND, chip_value
        elif sorted(rank_counts.values()) == [2, 3]:
            return HandType.FULL_HOUSE, chip_value
        elif is_flush:
            return HandType.FLUSH, chip_value
        elif is_straight:
            return HandType.STRAIGHT, chip_value
        elif 3 in rank_counts.values():
            return HandType.THREE_KIND, chip_value
        elif list(rank_counts.values()).count(2) == 2:
            return HandType.TWO_PAIR, chip_value
        elif 2 in rank_counts.values():
            return HandType.PAIR, chip_value
        else:
            return HandType.HIGH_CARD, chip_value

# do a super/sub class inheritance for organizational purposes
class Strategy:
    def __init__(self, name: str):
        self.name = name
    
    def decide_action(self, deck: List[Card], hand: List[Card], hands_left: int, discards_left: int) -> Tuple[str, List[Card]]:
        """Return ('play', cards_to_play) or ('discard', cards_to_discard)"""
        raise NotImplementedError

class HighCardStrategy(Strategy):
    def __init__(self):
        super().__init__("High Card Focus")
    
    def decide_action(self, deck: List[Card], hand: List[Card], hands_left: int, discards_left: int) -> Tuple[str, List[Card]]:
        # always play the 5 highest cards
        sorted_hand = sorted(hand, key=lambda c: c.rank.value, reverse=True)
        return ('play', sorted_hand[:5])

class PairHunterStrategy(Strategy):
    def __init__(self):
        super().__init__("Pair Hunter")
    
    def decide_action(self, deck: List[Card], hand: List[Card], hands_left: int, discards_left: int) -> Tuple[str, List[Card]]:
        rank_counts = Counter(card.rank for card in hand)
        
        # check if pair or better exists
        has_pair = any(count >= 2 for count in rank_counts.values())
        
        if has_pair or discards_left == 0:
            # play best 5 cards
            sorted_hand = sorted(hand, key=lambda c: (rank_counts[c.rank], c.rank.value), reverse=True)
            return ('play', sorted_hand[:5])
        else:
            # discard cards that don't contribute to pairs
            keep_cards = []
            for rank, count in rank_counts.items():
                cards_of_rank = [c for c in hand if c.rank == rank]
                if count > 1:
                    keep_cards.extend(cards_of_rank)
                else:
                    # keep highest single cards
                    if len(keep_cards) < 3:
                        keep_cards.extend(cards_of_rank)
            
            discard_cards = [c for c in hand if c not in keep_cards]
            return ('discard', discard_cards[:5])

class ConservativeStrategy(Strategy):
    def __init__(self):
        super().__init__("Conservative")
    
    def decide_action(self, deck: List[Card], hand: List[Card], hands_left: int, discards_left: int) -> Tuple[str, List[Card]]:
        sorted_hand = sorted(hand, key=lambda c: c.rank.value, reverse=True)
        best_5 = sorted_hand[:5]
        
        hand_type, _ = PokerEvaluator.evaluate_hand(best_5)
        
        # only play if we have pair or better, or if we're out of options
        if hand_type != HandType.HIGH_CARD or discards_left == 0 or hands_left == 1:
            return ('play', best_5)
        else:
            return ('discard', sorted_hand[5:])

class AggressiveStrategy(Strategy):
    def __init__(self):
        super().__init__("Aggressive")
    
    def decide_action(self, deck: List[Card], hand: List[Card], hands_left: int, discards_left: int) -> Tuple[str, List[Card]]:
        # almost always play unless we have a terrible hand and plenty of discards
        sorted_hand = sorted(hand, key=lambda c: c.rank.value, reverse=True)
        best_5 = sorted_hand[:5]
        
        # only discard if we have high cards < 7, lots of discards left
        if discards_left >= 2 and all(card.rank.value < 7 for card in best_5):
            return ('discard', sorted_hand[3:])
        else:
            return ('play', best_5)

"""
Flush and Straight both use the remaining cards in the deck as a means of making a decision
For instance if a hand has: 3 clubs, 3 hearts, 1 diamond, and 1 spade the diamond and 
spade will be discarded, but we discard clubs or hearts depending on which has fewer
remaining in the deck.
"""
class FlushChaserStrategy(Strategy):
    def __init__(self):
        super().__init__("Flush Chaser")
    
    def decide_action(self, deck: List[Card], hand: List[Card], hands_left: int, discards_left: int) -> Tuple[str, List[Card]]:
        if not hand:
            return ('play', [])

        # cards per suit in current hand
        hand_suit_counts = Counter(card.suit for card in hand)

        # cards per suit in current deck
        deck_suit_counts = Counter(card.suit for card in deck)

        # score each suit based on hand count and deck potential
        suit_scores = {}
        for suit, count_in_hand in hand_suit_counts.items():
            count_in_deck = deck_suit_counts.get(suit, 0)
            score = count_in_hand + 0.5 * count_in_deck  # weight deck support less than current hand
            suit_scores[suit] = (score, count_in_hand, count_in_deck)

        if not suit_scores:
            sorted_hand = sorted(hand, key=lambda c: c.rank.value, reverse=True)
            return ('play', sorted_hand[:5])

        # pick best suit to chase
        best_suit, (score, hand_count, deck_count) = max(suit_scores.items(), key=lambda x: x[1])

        # if flush is impossible (e.g. 3 in hand + 1 in deck = only 4 max), abort flush
        if hand_count + deck_count < 5:
            sorted_hand = sorted(hand, key=lambda c: c.rank.value, reverse=True)
            return ('play', sorted_hand[:5])

        suited_cards = [card for card in hand if card.suit == best_suit]
        off_suit_cards = [card for card in hand if card.suit != best_suit]

        # if we already have a flush
        if hand_count >= 5:
            suited_cards.sort(key=lambda c: c.rank.value, reverse=True)
            return ('play', suited_cards[:5])

        # if we're close and have enough discards to attempt it
        needed = 5 - hand_count
        if needed <= discards_left:
            return ('discard', off_suit_cards)

        # not worth chasing, fallback to high card value play
        sorted_hand = sorted(hand, key=lambda c: c.rank.value, reverse=True)
        return ('play', sorted_hand[:5])

class StraightBuilderStrategy(Strategy):
    def __init__(self):
        super().__init__("Straight Builder")

    def _get_rank_values(self, card: Card) -> List[int]:
        # include 1 for ace-low support (A = 14 and also 1)
        if card.rank.name == 'ACE':
            return [1, card.rank.rank_order]
        return [card.rank.rank_order]

    def _get_all_rank_values(self, cards: List[Card]) -> List[int]:
        values = []
        for card in cards:
            values.extend(self._get_rank_values(card))
        return sorted(set(values))

    def _find_best_straight_window(self, ranks: List[int], deck_ranks: Counter) -> Tuple[List[int], int]:
        best_window = []
        max_matches = -1

        # tries to find the best 5 card window that is most complete or completeable
        for low in range(1, 11):
            window = list(range(low, low + 5))
            in_hand = [r for r in window if r in ranks]
            missing = [r for r in window if r not in ranks and deck_ranks[r] > 0]

            matches = len(in_hand)
            potential = len(in_hand) + len(missing)

            if potential == 5 and matches > max_matches:
                best_window = window
                max_matches = matches

        return best_window, max_matches

    def decide_action(self, deck: List[Card], hand: List[Card], hands_left: int, discards_left: int) -> Tuple[str, List[Card]]:
        
        # get 5 best cards sorted by rank, if is a straight/straight flush play that
        sorted_hand = sorted(hand, key=lambda c: c.rank.rank_order, reverse=True)
        best_5 = sorted_hand[:5]
        hand_type, _ = PokerEvaluator.evaluate_hand(best_5)
        
        if hand_type in [HandType.STRAIGHT, HandType.STRAIGHT_FLUSH]:
            return ('play', best_5)

        # get ranks of current hand
        hand_ranks = self._get_all_rank_values(hand)

        # get ranks of current deck
        deck_ranks = Counter(self._get_all_rank_values(deck))

        # find best 5 card window, how many of those are already in are hand?
        best_window, in_hand_count = self._find_best_straight_window(hand_ranks, deck_ranks)

        # dont chase if no window exists or we have <= 2 cards for a straight in current hand
        if not best_window or in_hand_count < 3:
            return ('play', best_5)

        # determine which cards contribute to the best straight window
        straight_cards = [
            card for card in hand
            if any(rv in best_window for rv in self._get_rank_values(card))
        ]

        straight_cards = sorted(straight_cards, key=lambda c: c.rank.rank_order, reverse=True)

        # if we already have a straight, play it (after removing potential duplicates)
        if in_hand_count == 5:
            unique_straight_cards = list(OrderedDict((card.rank, card) for card in straight_cards).values())
            return ('play', unique_straight_cards)

        # if we are one away and can draw, gamble for straight
        if in_hand_count == 4 and discards_left >= 1:
            discard = [card for card in hand if card not in straight_cards]
            return ('discard', discard)

        # if we are two away and have multiple discards, gamble for straight
        if in_hand_count == 3 and discards_left >= 2:
            discard = [card for card in hand if card not in straight_cards]
            return ('discard', discard[:min(2, len(discard))])

        return ('play', best_5)

class HighValueStrategy(Strategy):
    def __init__(self):
        super().__init__("High Value")
    
    def decide_action(self, deck: List[Card], hand: List[Card], hands_left: int, discards_left: int) -> Tuple[str, List[Card]]:
        # define high value cards (10, J, Q, K, A)
        high_value_threshold = 10
        
        high_cards = [card for card in hand if card.rank.value >= high_value_threshold]
        low_cards = [card for card in hand if card.rank.value < high_value_threshold]
        
        # if we have 5+ high cards, play them
        if len(high_cards) >= 5:
            high_cards.sort(key=lambda c: c.rank.value, reverse=True)
            return ('play', high_cards[:5])
        
        # if we have 3-4 high cards and discards available, gamble for more highs
        elif len(high_cards) >= 3 and discards_left > 0 and len(low_cards) > 0:
            low_cards.sort(key=lambda c: c.rank.value)
            discard_count = min(len(low_cards), 5)
            return ('discard', low_cards[:discard_count])
        
        # else, play 5 highest value cards we have
        else:
            all_cards = sorted(hand, key=lambda c: c.rank.value, reverse=True)
            return ('play', all_cards[:5])

class BalancedStrategy(Strategy):
    def __init__(self):
        super().__init__("Balanced")
    
    def decide_action(self, deck: List[Card], hand: List[Card], hands_left: int, discards_left: int) -> Tuple[str, List[Card]]:
        sorted_hand = sorted(hand, key=lambda c: c.rank.value, reverse=True)
        best_5 = sorted_hand[:5]
        
        # evaluate current best 5 hand
        hand_type, _ = PokerEvaluator.evaluate_hand(best_5)
        
        # check for pairs
        rank_counts = Counter(card.rank for card in hand)
        has_pair = any(count >= 2 for count in rank_counts.values())
        
        # if we have a strong hand, play it
        if hand_type in [HandType.STRAIGHT, HandType.FLUSH, HandType.FULL_HOUSE, 
                        HandType.FOUR_KIND, HandType.STRAIGHT_FLUSH, HandType.ROYAL_FLUSH]:
            return ('play', best_5)
        
        # if we have a good hand, 
        elif hand_type in [HandType.THREE_KIND, HandType.TWO_PAIR]:
            # Good hand - play unless we have lots of resources and a weak three of a kind
            if hand_type == HandType.THREE_KIND and discards_left >= 2 and hands_left >= 2:
                # Check if three of a kind is low value
                three_kind_rank = max(rank_counts.values())
                if three_kind_rank == 3:
                    for rank, count in rank_counts.items():
                        if count == 3 and rank.value <= 6:
                            # Low three of a kind, try to improve
                            worst_cards = sorted_hand[5:]
                            return ('discard', worst_cards[:2])
            return ('play', best_5)
        
        # if we have pair, play unless we have >= discards/hands AND pair is low
        elif has_pair:
            pair_rank = None
            for rank, count in rank_counts.items():
                if count >= 2:
                    pair_rank = rank
                    break
            
            if (pair_rank and pair_rank.value <= 4 and 
                discards_left >= 2 and hands_left >= 2):
                return ('discard', sorted_hand[6:])
            else:
                return ('play', best_5)
        
        # high card only
        else:
            if discards_left > 0 and hands_left > 1:
                # if every card in best 5 is high, play it
                if all(card.rank.value >= 9 for card in best_5):
                    return ('play', best_5)
                
                # else discard at least 3 low cards
                else:
                    low_cards = [card for card in hand if card.rank.value <= 7]
                    if low_cards:
                        return ('discard', low_cards[:min(3, len(low_cards))])
                    else:
                        return ('play', best_5)
            else:
                return ('play', best_5)

class BalatroSimulation:
    def __init__(self, env: simpy.Environment, config: GameConfig, strategy: Strategy, action_records: List[ActionRecord]):
        self.env = env
        self.config = config
        self.strategy = strategy
        self.deck = self._create_deck()
        self.hand = []
        self.current_ante = 0
        self.current_blind = 0
        self.score = 0
        self.game_over = False
        self.wins = 0
        self.losses = 0
        self.action_records = action_records
        
    def _create_deck(self) -> List[Card]:
        deck = []
        for suit in Suit:
            for rank in Rank:
                deck.append(Card(rank, suit))
        return deck
    
    def _shuffle_deck(self):
        random.shuffle(self.deck)
    
    def _draw_cards(self, num_cards: int):
        for _ in range(min(num_cards, len(self.deck))):
            if len(self.hand) < 8:
                self.hand.append(self.deck.pop())
    
    def _calculate_score(self, cards: List[Card]) -> int:
        hand_type, chip_value = PokerEvaluator.evaluate_hand(cards)
        return (hand_type.chips + chip_value) * hand_type.mult
    
    def _format_cards(self, cards: List[Card]) -> str:
        sorted_cards = sorted(cards, key=lambda c: c.rank.rank_order, reverse=False)
        return ", ".join(str(card) for card in sorted_cards)
    
    def run_blind(self, blind_config: BlindConfig, ante_num: int, game_num: int) -> bool:
        hands_left = self.config.hands_per_blind
        discards_left = self.config.discards_per_blind
        total_score = 0

        print(f"  {blind_config.name} Blind (Target: {blind_config.target_score})")
        
        # initial hand
        self._draw_cards(8)

        while hands_left > 0:
            # use specific decide_action method depending on current strategy
            action, cards = self.strategy.decide_action(self.deck, self.hand, hands_left, discards_left)
            
            if action == 'play':
                hands_left -= 1
                score = self._calculate_score(cards)
                total_score += score
                hand_type, _ = PokerEvaluator.evaluate_hand(cards)

                print(f"    Played {self._format_cards(cards)} ({hand_type.name}): {score} points (total = {total_score})")
                
                blind_defeated = "Yes" if total_score >= blind_config.target_score else "No"
                
                record = ActionRecord(
                    game_number=game_num,
                    strategy_name=self.strategy.name,
                    config_name=self.config.config_name,
                    ante=ante_num,
                    blind_name=blind_config.name,
                    current_deck_size=len(self.deck),
                    current_hand=self._format_cards(self.hand),
                    action="Play",
                    cards_played=self._format_cards(cards),
                    hand_type=hand_type.handName,
                    score_for_play=score,
                    total_score_for_blind=total_score,
                    hands_remaining=hands_left,
                    discards_remaining=discards_left,
                    blind_defeated=blind_defeated
                )
                self.action_records.append(record)
                
                if total_score >= blind_config.target_score:
                    print(f"    ✓ Blind defeated!")
                    return True, total_score
                
                # remove played cards from hand
                for card in cards:
                    if card in self.hand:
                        self.hand.remove(card)
                
                # draw back to 8 cards
                self._draw_cards(8 - len(self.hand))
                
            elif action == 'discard':
                if discards_left > 0:
                    discards_left -= 1
                    
                    record = ActionRecord(
                        game_number=game_num,
                        strategy_name=self.strategy.name,
                        config_name=self.config.config_name,
                        ante=ante_num,
                        blind_name=blind_config.name,
                        current_deck_size=len(self.deck),
                        current_hand=self._format_cards(self.hand),
                        action="Discard",
                        cards_played=self._format_cards(cards),
                        hand_type="",
                        score_for_play=0,
                        total_score_for_blind=total_score,
                        hands_remaining=hands_left,
                        discards_remaining=discards_left,
                        blind_defeated="No"
                    )
                    self.action_records.append(record)
                    
                    # remove discarded cards (temporarily)
                    for card in cards:
                        if card in self.hand:
                            self.hand.remove(card)
                    
                    # draw back to 8 cards
                    self._draw_cards(8 - len(self.hand))

                    # add discarded cards back into deck
                    for card in cards:
                        self.deck.append(card)

                    # reshuffle just in case
                    self._shuffle_deck()

                else:
                    # no discards left, must play
                    hands_left -= 1
                    score = self._calculate_score(cards)
                    total_score += score
                    hand_type, _ = PokerEvaluator.evaluate_hand(cards)
                    
                    blind_defeated = "Yes" if total_score >= blind_config.target_score else "No"
                    
                    record = ActionRecord(
                        game_number=game_num,
                        strategy_name=self.strategy.name,
                        config_name=self.config.config_name,
                        ante=ante_num,
                        blind_name=blind_config.name,
                        current_deck_size=len(self.deck),
                        current_hand=self._format_cards(self.hand),
                        action="Forced Play",
                        cards_played=self._format_cards(cards),
                        hand_type=hand_type.handName,
                        score_for_play=score,
                        total_score_for_blind=total_score,
                        hands_remaining=hands_left,
                        discards_remaining=discards_left,
                        blind_defeated=blind_defeated
                    )
                    self.action_records.append(record)
                    
                    if total_score >= blind_config.target_score:
                        return True, total_score
        
        return False, total_score
    
    def run_game(self, game_num: int) -> bool:
        print(f"\n=== New Game: {self.strategy.name} ===")

        total_game_score = 0
        
        for ante_num, ante_blinds in enumerate(self.config.antes):

            print(f"\nAnte {ante_num + 1}:")
            
            for blind_config in ante_blinds:
                # reset deck/hand for each blind
                self.deck = self._create_deck()
                self._shuffle_deck()
                self.hand = []

                success, total_blind_score = self.run_blind(blind_config, ante_num, game_num)
                total_game_score += total_blind_score
                if not success:
                    print(f"Game Over at Ante {ante_num + 1}. Final Game Score = {total_game_score}")
                    self.losses += 1
                    return False, total_game_score
        
        print(f"Game Won! Completed all antes. Final Game Score = {total_game_score}")
        self.wins += 1
        return True, total_game_score


"""
Game configurations are based on a combination of actual Balatro modes:
- Blue Deck: 5 hands and 4 discards per blind
- White, Green, and Purple Stakes: https://balatrowiki.org/w/Stakes
- Small Blinds: https://balatrowiki.org/w/Small_Blind
- Big Blinds: https://balatrowiki.org/w/Big_Blind
- Boss Blinds: a lot of bosses use these targets
"""

EASY_CONFIG = GameConfig(
    config_name="White",
    hands_per_blind=5,
    discards_per_blind=4,
    antes=[
        [BlindConfig("Small", 300), BlindConfig("Big", 450), BlindConfig("Boss", 600)],
        [BlindConfig("Small", 800), BlindConfig("Big", 1200), BlindConfig("Boss", 1600)],
        [BlindConfig("Small", 2000), BlindConfig("Big", 3000), BlindConfig("Boss", 4000)]
    ]
)

NORMAL_CONFIG = GameConfig(
    config_name="Green",
    hands_per_blind=4,
    discards_per_blind=5,
    antes=[
        [BlindConfig("Small", 300), BlindConfig("Big", 450), BlindConfig("Boss", 600)],
        [BlindConfig("Small", 900), BlindConfig("Big", 1350), BlindConfig("Boss", 1800)],
        [BlindConfig("Small", 2600), BlindConfig("Big", 3900), BlindConfig("Boss", 5200)]
    ]
)

HARD_CONFIG = GameConfig(
    config_name="Purple",
    hands_per_blind=4,
    discards_per_blind=4,
    antes=[
        [BlindConfig("Small", 300), BlindConfig("Big", 450), BlindConfig("Boss", 600)],
        [BlindConfig("Small", 1000), BlindConfig("Big", 1500), BlindConfig("Boss", 2000)],
        [BlindConfig("Small", 3200), BlindConfig("Big", 4800), BlindConfig("Boss", 6400)]
    ]
)

def run_simulation(num_games: int = 100):
    strategies = [
        HighCardStrategy(),
        PairHunterStrategy(),
        ConservativeStrategy(),
        AggressiveStrategy(),
        FlushChaserStrategy(),
        StraightBuilderStrategy(),
        HighValueStrategy(),
        BalancedStrategy()
    ]
    
    configs = [
        ("Easy", EASY_CONFIG),
        ("Normal", NORMAL_CONFIG),
        ("Hard", HARD_CONFIG)
    ]
    
    results = {}
    all_action_records = {}
    all_game_actions = []
    
    for config_name, config in configs:
        print(f"\n{'='*50}")
        print(f"TESTING CONFIGURATION: {config_name}")
        print(f"{'='*50}")
        
        config_results = {}
        
        for strategy in strategies:
            print(f"\nTesting {strategy.name} strategy...")
            
            scores = []
            games_played = 0
            
            for game_num in range(num_games):
                env = simpy.Environment()
                sim = BalatroSimulation(env, config, strategy, all_game_actions)
                
                # Run single game and get the final score
                is_game_win, game_score = sim.run_game(game_num) 
                scores.append(game_score)
                games_played += 1

                # add action records for game to master list
                all_game_actions = sim.action_records
                
                # Show progress every 20 games
                if (game_num + 1) % 20 == 0:
                    avg_score = sum(scores) / len(scores)
                    ci_lower, ci_upper = calculate_score_confidence_interval(scores)
                    print(f"  Progress: {game_num + 1}/{num_games} games, "
                          f"Avg score: {avg_score:.0f} [95% CI: {ci_lower:.0f}-{ci_upper:.0f}]")
            
            avg_score = sum(scores) / len(scores)
            ci_lower, ci_upper = calculate_score_confidence_interval(scores)
            
            config_results[strategy.name] = {
                'scores': scores,
                'games': games_played,
                'avg_score': avg_score,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'total_score': sum(scores)
            }
            
            print(f"  Final: Avg score {avg_score:.0f} over {games_played} games "
                  f"[95% CI: {ci_lower:.0f}-{ci_upper:.0f}]")
            
            # add all actions to overall dictionary
            all_action_records[strategy.name] = [action for action in all_game_actions if action.strategy_name == strategy.name]
        
        results[config_name] = config_results

    # write actions to excel
    save_to_excel(all_action_records)

    # print summary with statistcal analysis
    print_statistical_results(num_games, results)
    
    return results

# run the simulation here
if __name__ == "__main__":
    # set random seed for reproducible results
    random.seed(123)
    
    print("Starting Balatro Strategy Simulation...")
    print("Testing different strategies across multiple difficulty levels...")
    
    results = run_simulation(500)