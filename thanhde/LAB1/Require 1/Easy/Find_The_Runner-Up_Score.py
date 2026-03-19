n = int(input())
arr = list(map (int, input().split()))
result = sorted(arr)
for i in range(n - 1, -1, -1):
    if result[i - 1] != result[i]:
        print(result[i - 1])
        break