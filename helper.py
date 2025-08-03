import math
from typing import List, Tuple, Dict
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import os

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

def calculate_score_confidence_interval(scores: list, confidence: float = 0.95) -> Tuple[float, float]:
    if len(scores) == 0:
        return (0.0, 0.0)
    
    n = len(scores)
    mean_score = sum(scores) / n
    
    if n == 1:
        return (mean_score, mean_score)
    
    variance = sum((x - mean_score) ** 2 for x in scores) / (n - 1)
    std_dev = math.sqrt(variance)
    
    se = std_dev / math.sqrt(n)
    
    if confidence == 0.95:
        t_critical = 1.96 if n >= 30 else 2.0 
    elif confidence == 0.99:
        t_critical = 2.576 if n >= 30 else 2.6
    else:
        t_critical = 1.645 if n >= 30 else 1.7
    
    margin = t_critical * se
    lower = mean_score - margin
    upper = mean_score + margin
    
    return (lower, upper)

def statistical_significance_test_scores(scores1: list, scores2: list) -> Tuple[float, bool]:
    if len(scores1) == 0 or len(scores2) == 0:
        return (0.0, False)
    
    n1, n2 = len(scores1), len(scores2)
    mean1 = sum(scores1) / n1
    mean2 = sum(scores2) / n2
    
    if n1 == 1 and n2 == 1:
        return (0.0, False)
    
    var1 = sum((x - mean1) ** 2 for x in scores1) / (n1 - 1) if n1 > 1 else 0
    var2 = sum((x - mean2) ** 2 for x in scores2) / (n2 - 1) if n2 > 1 else 0
    
    pooled_se = math.sqrt(var1/n1 + var2/n2)
    
    if pooled_se == 0:
        return (0.0, False)
    
    t_score = (mean1 - mean2) / pooled_se
    is_significant = abs(t_score) > 1.96
    
    return (t_score, is_significant)

def print_statistical_results(num_games: int, results: dict):
    print(f"\n{'='*80}")
    print("MONTE CARLO SIMULATION RESULTS - STATISTICAL ANALYSIS")
    print(f"{'='*80}")
    print(f"Sample size: {num_games} games per strategy per configuration")
    print("Confidence intervals: 95% confidence intervals for mean scores")
    
    for config_name, config_results in results.items():
        print(f"\n{config_name} Configuration:")
        print("-" * 60)
        
        sorted_strategies = sorted(config_results.items(), 
                                 key=lambda x: x[1]['avg_score'], 
                                 reverse=True)
        
        print(f"{'Rank':<4} {'Strategy':<20} {'Avg Score':<15} {'95% CI':<25} {'Games'}")
        print("-" * 60)
        
        for rank, (strategy_name, stats) in enumerate(sorted_strategies, 1):
            ci_str = f"[{stats['ci_lower']:.0f}-{stats['ci_upper']:.0f}]"
            print(f"{rank:<4} {strategy_name:<20} {stats['avg_score']:>8.0f} {' '*7} "
                  f"{ci_str:<25} {stats['games']}")
        
        if len(sorted_strategies) >= 2:
            print(f"\nStatistical Significance Tests (α = 0.05):")
            print("-" * 60)
            
            best_strategy = sorted_strategies[0]
            best_name, best_stats = best_strategy
            
            print(f"Comparing {best_name} (best) against other strategies:")
            
            for other_name, other_stats in sorted_strategies[1:4]:
                t_score, is_sig = statistical_significance_test_scores(
                    best_stats['scores'], other_stats['scores']
                )
                
                diff = best_stats['avg_score'] - other_stats['avg_score']
                sig_indicator = "***" if is_sig else "n.s."
                
                print(f"  vs {other_name:<20}: {diff:+8.0f} score difference, "
                      f"t={t_score:+5.2f} {sig_indicator}")
            
            print("  *** = Statistically significant at p < 0.05")
            print("  n.s. = Not statistically significant")
    
    print(f"\n{'='*80}")
    print("CROSS-CONFIGURATION ANALYSIS")
    print(f"{'='*80}")
    
    all_strategy_names = set()
    for config_results in results.values():
        all_strategy_names.update(config_results.keys())
    
    strategy_averages = {}
    for strategy_name in all_strategy_names:
        all_scores = []
        total_games = 0
        for config_results in results.values():
            if strategy_name in config_results:
                stats = config_results[strategy_name]
                all_scores.extend(stats['scores'])
                total_games += stats['games']
        
        if total_games > 0:
            avg_score = sum(all_scores) / len(all_scores)
            ci_lower, ci_upper = calculate_score_confidence_interval(all_scores)
            strategy_averages[strategy_name] = {
                'avg_score': avg_score,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'all_scores': all_scores,
                'total_games': total_games
            }
    
    print("Overall Performance Rankings (across all configurations):")
    print("-" * 70)
    print(f"{'Rank':<4} {'Strategy':<20} {'Avg Score':<15} {'95% CI':<25} {'Total Games'}")
    print("-" * 70)
    
    sorted_overall = sorted(strategy_averages.items(), 
                           key=lambda x: x[1]['avg_score'], 
                           reverse=True)
    
    for rank, (strategy_name, stats) in enumerate(sorted_overall, 1):
        ci_str = f"[{stats['ci_lower']:.0f}-{stats['ci_upper']:.0f}]"
        print(f"{rank:<4} {strategy_name:<20} {stats['avg_score']:>8.0f} {' '*7} "
              f"{ci_str:<25} {stats['total_games']}")

def save_to_excel(all_action_records: Dict[str, List[ActionRecord]], filename: str = "balatro_simulation_data.xlsx"):
    wb = Workbook()
    wb.remove(wb.active)
    
    for strategy_name, records in all_action_records.items():
        print("writing strategy_name:", strategy_name)
        data = []
        for record in records:
            data.append({
                'Game Number': record.game_number,
                'Config Name': record.config_name,
                'Ante': record.ante,
                'Blind Name': record.blind_name,
                'Current Deck Size': record.current_deck_size,
                'Current Hand': record.current_hand,
                'Action': record.action,
                'Cards Played': record.cards_played,
                'Hand Type': record.hand_type,
                'Score for Play': record.score_for_play,
                'Total Score for Blind': record.total_score_for_blind,
                'Hands Remaining': record.hands_remaining,
                'Discards Remaining': record.discards_remaining,
                'Blind Defeated': record.blind_defeated
            })
        
        if data:
            df = pd.DataFrame(data)
            ws = wb.create_sheet(title=strategy_name.replace(' ', '_'))
            
            for r in dataframe_to_rows(df, index=False, header=True):
                ws.append(r)
            
            # adjust column widths
            for column in ws.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                ws.column_dimensions[column_letter].width = adjusted_width

    wb.save(filename)
    print(f"Action data saved to {filename}")