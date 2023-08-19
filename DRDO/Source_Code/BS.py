import numpy as np

arr = [1,4,2,5,3,0]

for i in range(len(arr)):
    for j in range(i,len(arr)-1):
        if arr[j]>arr[j+1]:
            temp = arr[j+1]
            arr[j+1] = arr[j]
            arr[j] = temp

print(arr)