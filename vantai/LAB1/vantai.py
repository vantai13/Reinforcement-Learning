if __name__ == '__main__': 
    
    n, m = map(int, input().split())
    arr_n = list(map(int, input().split()))
    arr_A = set(list(map(int, input().split())))
    arr_B = set(list(map(int, input().split())))

    happiness = 0

    for value in arr_n:
        if value in arr_A: 
            happiness += 1
        if value in arr_B: 
            happiness -= 1

    

    print(happiness)