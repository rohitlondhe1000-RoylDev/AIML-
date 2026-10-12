import statistics

numbers = [11, 22, 33, 44, 55]

print("Numbers:", numbers)

total = sum(numbers)
average = total / len(numbers)
minimum = min(numbers)
maximum = max(numbers)

print("\n--- Arithmetic Calculations ---")
print(f"Sum     : {total}")
print(f"Average : {average:.2f}")
print(f"Minimum : {minimum}")
print(f"Maximum : {maximum}")

median_value = statistics.median(numbers)
std_deviation = statistics.stdev(numbers)

print("\n--- Statistical Calculations ---")
print(f"Mean               : {average:.2f}")
print(f"Median             : {median_value:.2f}")
print(f"Standard Deviation : {std_deviation:.2f}")