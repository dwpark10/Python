#!/usr/bin/env python3
# =*- coding: UTF-8 -*-

import sys
import os

# Data list
equip_list = []

# 0. 메뉴 출력
def print_menu():
    print("1. 입력")
    print("2. 출력")
    print("3. 수정")
    print("4. 검색")
    print("5. 삭제")
    print("6. 정렬")
    print("7. 저장")
    print("8. 종료")
    menu = int(input("메뉴를 선택하시오:"))
    return menu

# 1. 데이터 입력
def data_input(arg = 0):
    if arg == 0:
        equip_name = input("장비명 : ")
        equip_quantity = int(input("수량 : "))
        equip_date = input("생산일 : ")
        equip_remain = input("재고여부(예:y or n) : ")
        temp = input("계속입력하시겠습니까?(y/n)")
        res = True if ( temp == 'y' or temp == 'Y') else False
        my_data = {"name":equip_name, "quantity":equip_quantity, "date":equip_date, "remain":equip_remain}
        equip_list.append(my_data)
        if res is True:
            data_input()
        else:
            pass
    else:
        equip_list.append(arg)

# 2. 데이터 출력
def data_print(arg = 0):
    if arg is 0:
        print("--------------------------------------------------------")
        print("    장비명          수량         생산일       재고여부  ")
        print("--------------------------------------------------------")
        for my in equip_list:
            print("%10s\t%7d\t%16s\t%3s" % (my["name"], my["quantity"], my["date"], my["remain"]))
    else:
        print("--------------------------------------------------------")
        print("    장비명          수량         생산일       재고여부  ")
        print("--------------------------------------------------------")
        print("%10s\t%7d\t%16s\t%3s" % (arg["name"], arg["quantity"], arg["date"], arg["remain"]))

# 3. 데이터 수정
def data_modify(arg = 0):
    dat = input("수정할 장비명을 입력하세요 : ")
    res = data_search(dat)
    if res is not None:
        res["name"] = input("장비명("+res["name"]+")")
        res["quantity"] = int(input("수량("+str(res["quantity"])+")"))
        res["date"] = input("생산일("+res["date"]+")")
        res["remain"] = input("재고여부("+res["remain"]+")")
    else:
        print("해당 장비가 없습니다")

# 4. 데이터 검색
def data_search(arg = 0):
    res = None
    if arg is 0:
        dat = input("검색할 장비명을 입력하세요 : ")
    else:
        dat = arg

    for my in equip_list:
        if my["name"] == dat:
            res = my

    if arg is 0:
        if res is not None:
            data_print(res)
        else:
            print("해당 장비가 없습니다")
    else:
        return res

# 5. 데이터 삭제
def data_delete(arg = 0):
    dat = input("삭제할 장비명을 입력하세요 : ")
    res = data_search(dat)
    if res is not None:
        equip_list.remove(res)
    else:
        print("해당 장비가 없습니다")

# 6. 데이터 정렬
def data_sort(arg = 0):
    equip_list.sort(key=lambda equip : equip["name"])
    data_print()

# 7. 데이터 저장
def data_save(arg = 0):
    try:
        fp = open(os.path.join(os.getcwd(), "data.txt"), "w")
        for my in equip_list:
            fp.write(my["name"]+","+str(my["quantity"])+","+my["date"]+","+my["remain"]+"\n")
        fp.close
    except IOError as e:
        pass

# 8. 종료
def prog_exit(arg = 0):
    res = input("프로그램을 종료하시겠습니까(y/n)?")
    if res is 'y' or res is 'Y':
        sys.exit(1)
    else:
        pass


case_dict = {1:data_input, 2:data_print, 3:data_modify, 4:data_search, 5:data_delete, 6:data_sort, 7:data_save, 8:prog_exit}

def DoProcess():
    try:
        fp = open(os.path.join(os.getcwd(), "input.txt"), "r")
        while True:
            line = fp.readline()
            if not line: break
            line = line[:-1]
            temp_list = line.split(',')
            equip_data = {"name":temp_list[0], "quantity":int(temp_list[1]), "date":temp_list[2], "remain":temp_list[3]}
            data_input(equip_data)
        fp.close()
    except IOError as e:
        pass

    while( 1 ):
        menu = print_menu()
        if 1 <= menu <= 8:
            case_dict[menu](0)
        else:
            print("잘못 입력하였습니다")






#main
DoProcess()


