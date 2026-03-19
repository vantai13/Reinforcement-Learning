if __name__ == '__main__':
    n = int(input())
    lists = []
    for i in range(0, n):
        str = input().split()
        if (str[0] == "insert"):
            lists.insert(int(str[1]), int(str[2]))
        elif (str[0] == "print"):
            print(lists)
        elif (str[0] == "remove"):
            lists.remove(int(str[1]))
        elif (str[0] == "append"):
            lists.append(int(str[1]))
        elif (str[0] == "sort"):
            lists.sort()
        elif (str[0] == "pop"):
            lists.pop()
        elif (str[0] == "reverse"):
            lists.reverse()
        else: break