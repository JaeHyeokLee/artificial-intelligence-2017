#-*- coding: utf-8 -*-

# 5*5 배열 선언을 위해서 numpy 패키지 이용
# import numpy as np 와 같이 선언한다.

import numpy as np
import Tkinter as tk

#전체 지도 - 5*5 배열 선언
search_map = np.zeros(shape = (5,5))

#막힌 곳의 값을 -1로 설정

search_map[0][3] = -1
search_map[1][1] = -1
search_map[2][1] = -1
search_map[1][3] = -1
search_map[2][3] = -1
search_map[4][3] = -1


# a* search 알고리즘을 구현하기 위한 H(x) 값을 구해준다.
# 5*5의 array이므로 index는 0부터 4까지 존재.
# Horizental Distance 와 Vertical Distance 를 더해준 값

def H(cur_x, cur_y):
	return abs(4 - cur_x) + abs(2 - cur_y)


#다음 이동할 방향으로 이동 가능한지 확인해주는 함수
#possible_list = [오른쪽, 왼쪽, 위쪽, 아래쪽]
#이동 가능하지 않은 경우(격자 밖으로 나가는 경우, 해당 길이 막혀있는 경우) 해당 위치에 -1을 넣어준다.

def is_movable(cur_x, cur_y):
	possible_list = [1,1,1,1]

	if ((cur_x == 4) or (search_map[cur_y][cur_x+1] < 0)) == True:
		possible_list[0] = -1
	if ((cur_x == 0) or (search_map[cur_y][cur_x-1] < 0)) == True:
		possible_list[1] = -1
	if ((cur_y == 4) or (search_map[cur_y+1][cur_x] < 0)) == True:
		possible_list[2] = -1
	if ((cur_y == 0) or (search_map[cur_y-1][cur_x] < 0)) == True:
		possible_list[3] = -1

	return possible_list

#현재 위치에서의 H(x), G(x), F(x)를 구하고, F(x)가 최소가 되는 위치로 이동한다.

def move(cur_x, cur_y, count):
	possible_list = is_movable(cur_x, cur_y)
	for i in range(len(possible_list)):
		if possible_list[i]>0:
			if i == 0:
				possible_list[i] = H(cur_x+1, cur_y) + count
			elif i == 1:
				possible_list[i] = H(cur_x-1, cur_y) + count
			elif i == 2:
				possible_list[i] = H(cur_x, cur_y+1) + count
			elif i == 3:
				possible_list[i] = H(cur_x, cur_y-1) + count
		else:
			possible_list[i] = 999

		#이동가능한 경우, possible_list 라는 리스트에 H(x) + G(x) 의 값을 넣어준다. ( = F(x))
		#possible_list 에서 값이 최소인 원소의 인덱스를 반환할 예정이므로, 이동이 불가능한 경우 999를 넣어준다.

		#최소인 값을 찾아서 위치 반환
	direction = possible_list.index(min(possible_list))	
		
	if direction == 0:
		cur_x = cur_x + 1
	elif direction == 1:
		cur_x = cur_x - 1
	elif direction == 2:
		cur_y = cur_y + 1
	elif direction == 3:
		cur_y = cur_y - 1

	return [cur_x, cur_y]


#콘솔에 현재 위치와 현재 위치에서의 H(x), G(x), F(x)의 갑을 출력해준다.

def print_console(cur_x, cur_y, count):
	print(str(count) + ' : '  + ' (' + str(cur_x) + ', ' + str(cur_y) + ') -> ' + str(cur_x + 5 * cur_y + 1) + '\n')
	print('H(x) : ' + str(H(cur_x, cur_y)) + ', G(x) : ' + str(count) + ', F(x) : ' + str(H(cur_x, cur_y) + count) + '\n')

#Tkinter App Class
class App:
	def __init__(self, master):
		#미로를 그릴 캔버스 생성

		self.canvas = tk.Canvas(master, width = 800, height = 600)
		self.canvas.pack()

		#Run 버튼 생성
		button = tk.Button(master, text = 'run', command = self.run)
		button.pack()

		#전체 미로 그리기

		for col in range(5):
			for row in range(5):
				if search_map[row][col] == 0:
					fillColor = 'white'
				else:
					fillColor = 'black'

				self.canvas.create_rectangle(col * 100, row * 100, col * 100 + 100, row * 100 + 100, fill = fillColor, outline = 'blue')
				self.canvas.create_text(col * 100 + 50, row * 100 + 50, text = col + 5 * row + 1)

		self.start = [0,2]
		self.count = 0
		self.route = []
	#Run 버튼을 눌렀을 시 실행되는 함수 정의
	def run(self):

			col = self.start[0]
			row = self.start[1]
			self.route.append(col + 5 * row + 1)
				
				
			print_console(col, row, self.count)
			if(col == 4 and row == 2):
				print('Count : ' + str(self.count) + '\n')
				print('Route' + '\n')
				print(self.route)	

				self.canvas.create_text(600, 200, text = '이동횟수 : ' + str(self.count))
				self.canvas.create_text(600, 225, text = '경로')
				self.canvas.create_text(600, 250, text = self.route)

			self.canvas.create_rectangle(col * 100, row * 100, col * 100 + 100, row * 100 + 100, fill = 'red', outline = 'blue')
			self.start = move(self.start[0], self.start[1], self.count)
			self.count = self.count + 1

#Tkinter 패키지를 이용한 GUI Loop

root = tk.Tk()
app = App(root)
root.mainloop()