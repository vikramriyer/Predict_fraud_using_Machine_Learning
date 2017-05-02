def featureScaling(arr):
    max_value = max(arr)
    min_value = min(arr)
    desired_value = [i for i in arr if i != max_value and i != min_value][0]
    value = (desired_value - min_value)/float(max_value - min_value)
    return value

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)
