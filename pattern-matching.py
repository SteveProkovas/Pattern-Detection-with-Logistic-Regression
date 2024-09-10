def find_pattern(sequence, pattern):
    pattern_length = len(pattern)
    correct_positions = []
    incorrect_positions = []
    
    for i in range(len(sequence) - pattern_length + 1):
        if sequence[i:i + pattern_length] == pattern:
            correct_positions.append(f'P{i + 1}')
        else:
            incorrect_positions.append(f'P{i + 1}')
    
    return correct_positions, incorrect_positions

# Example usage
sequence = [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4]
pattern = [1, 2, 3, 4]

correct_positions, incorrect_positions = find_pattern(sequence, pattern)

print(f"Correct patterns found at: {', '.join(correct_positions) if correct_positions else 'None'}")
print(f"Incorrect positions: {', '.join(incorrect_positions) if incorrect_positions else 'None'}")
