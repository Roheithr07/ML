import pandas as pd

data = pd.read_csv("student_data.csv")

# Use existing columns from your dataset
internal_mark = data["studytime"]
attendance = 100 - data["absences"]
final_mark = data["failures"]

print("Internal Mark")
print("Mean =", round(internal_mark.mean(), 3))
print("Median =", round(internal_mark.median(), 3))
print("Mode =", round(internal_mark.mode().iloc[0], 3))
print("Standard Deviation =", round(internal_mark.std(), 3))

print("\nAttendance")
print("Mean =", round(attendance.mean(), 3))
print("Median =", round(attendance.median(), 3))
print("Mode =", round(attendance.mode().iloc[0], 3))
print("Standard Deviation =", round(attendance.std(), 3))

print("\nFinal Mark")
print("Mean =", round(final_mark.mean(), 3))
print("Median =", round(final_mark.median(), 3))
print("Mode =", round(final_mark.mode().iloc[0], 3))
print("Standard Deviation =", round(final_mark.std(), 3))

