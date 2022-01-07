import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as path
import math

def dist(x1,y1,x2,y2):
	return math.sqrt((x1-x2)*(x1-x2)+ (y1-y2)*(y1-y2))

grid_x = np.arange(101)
grid_y = np.arange(101)

plt.xlim(0,100)
plt.ylim(0,100)

#example 1
start_x1 = 1
start_y1 = 3

goal_x1 = 53
goal_y1 = 75

#example 2
start_x2 = 92
start_y2 = 12

goal_x2 = 20
goal_y2 = 75

#example 3
start_x3 = 32
start_y3 = 41

goal_x3 = 91
goal_y3 = 4

#currently used example
start_x = start_x1
start_y = start_y1

goal_x = goal_x1
goal_y = goal_y1

plt.plot(start_x,start_y,'.')
plt.text(start_x+0.5,start_y,"start")
plt.plot(goal_x,goal_y,'.')
plt.text(goal_x+0.5,goal_y,"goal")

#example 1
obstacle_vertices1 = np.array([[44,26],[23,64],[73,12]])

#example 2
obstacle_vertices2 = np.array([[32,21],[63,61],[33,65],[3,12]])

#example 3
obstacle_vertices3 = np.array([[22,19],[39,25],[7,33]])


#currently used obstacle
obstacle_vertices = obstacle_vertices3

obstacle = path.Path(obstacle_vertices)

path_false = obstacle.contains_points([[1000000,1000000]])


obstacle_x = []
obstacle_y = []

for i in range(len(obstacle_vertices)):
	obstacle_x.append(obstacle_vertices[i][0])
	obstacle_y.append(obstacle_vertices[i][1])

obstacle_x.append(obstacle_vertices[0][0])
obstacle_y.append(obstacle_vertices[0][1])

obstacle_x = np.array(obstacle_x)
obstacle_y = np.array(obstacle_y)

plt.plot(obstacle_x, obstacle_y)

plt.plot([start_x,goal_x], [start_y, goal_y], 'rs--',  label='m-line')

plt.grid()
plt.legend()

m_line = path.Path(np.array([[start_x,start_y],[goal_x,goal_y]]))

#rotate by n
angle = math.pi / 4
rotation_matrix = np.array([[math.cos(angle),-1*math.sin(angle)],[math.sin(angle),math.cos(angle)]])

reverse_angle = 2*math.pi - angle
reverse_rotation_matrix = np.array([[math.cos(reverse_angle),-1*math.sin(reverse_angle)],[math.sin(reverse_angle),math.cos(reverse_angle)]])

#start of algorithm
current_x = start_x
current_y = start_y

stride = 2

while current_x != goal_x or current_y != goal_y:
	obstacles_hit = []
	next_x = current_x
	next_y = current_y

	if dist(current_x,current_y,goal_x,goal_y) < stride+1:
		next_x = goal_x
		next_y = goal_y
		plt.plot([current_x,next_x],[current_y,next_y])
		current_x = next_x
		current_y = next_y
		break

	x_vect = stride * (goal_x - current_x) / dist(current_x,current_y,goal_x,goal_y)
	y_vect = stride * (goal_y - current_y) / dist(current_x,current_y,goal_x,goal_y)

	next_x = current_x + x_vect
	next_y = current_y + y_vect

	if obstacle.contains_points([[next_x,next_y]]): #Encountered obstacle, starting Head
		last_head = [current_x,current_y]
		vector = np.array([x_vect,y_vect])
		while obstacle.contains_points([[next_x,next_y]]) != 0:
			vector = np.matmul(vector,rotation_matrix)
			next_x = current_x + vector[0]
			next_y = current_y + vector[1]
		plt.plot(next_x,next_y,'.')
		plt.plot([current_x,next_x],[current_y,next_y])

		x_vect = stride * (next_x - current_x) / dist(current_x,current_y,next_x,next_y)
		y_vect = stride * (next_y - current_y) / dist(current_x,current_y,next_x,next_y)
		current_x = next_x
		current_y = next_y
		
		while 1== 1:
			vector = np.array([x_vect,y_vect])
			if obstacle.contains_points([[current_x+vector[0],current_y+vector[1]]])!= path_false:#when going forwards immediately encounters an obstacle
				while 1==1:
					vector_temp = np.matmul(vector, rotation_matrix)
					if obstacle.contains_points([[current_x+vector_temp[0],current_y+vector_temp[1]]]) == path_false:
						vector = vector_temp
						break
					vector = vector_temp
			else:
				tries = 0
				orig_vector = vector
				while 1==1:
					vector_temp = np.matmul(vector, reverse_rotation_matrix)
					if obstacle.contains_points([[current_x+vector_temp[0],current_y+vector_temp[1]]]) != path_false:
						break
					if tries > 2*math.pi/angle:
						vector = orig_vector
						break
					vector = vector_temp
					tries+=1

			next_x += vector[0]
			next_y += vector[1]

			current_path = path.Path(np.array([[current_x,current_y],[next_x,next_y]]))
			if m_line.intersects_path(current_path): #Encountered m-line, starting Leave
				if goal_x == start_x:#when m-line is across y axis
					slope_m = 100000 #aproximate infinity
				else:
					slope_m = float((goal_y-start_y))/float(goal_x-start_x)

				if next_x == current_x:
					slope_c = 100000
				else:
					slope_c = (next_y-current_y)/(next_x-current_x)
				intercept_m = goal_y - slope_m*goal_x
				intercept_c = next_y - slope_c*next_x

				x_final = (-1*(intercept_c-intercept_m))/(slope_c-slope_m)
				y_final = slope_m*x_final + intercept_m

				next_x = x_final
				next_y = y_final
				break

			plt.plot(next_x,next_y,'.')
			plt.plot([current_x,next_x],[current_y,next_y])

			if dist(next_x,next_y,last_head[0], last_head[1]) < (stride - 0.1):
				plt.show()
				print("Failed to find goal")
				quit()

			current_x = next_x
			current_y = next_y


	plt.plot(next_x,next_y,'.')
	plt.plot([current_x,next_x],[current_y,next_y])
	current_x = next_x
	current_y = next_y

plt.show()
