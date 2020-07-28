# ####################################################
# DE2-COM2 Computing 2
# Individual project
#
# Title: MAIN
# Authors: Nour Moussa
# ####################################################

import random 
from copy import deepcopy #Deep copy: the copying process occurs recursively populating it with copies of the child objects found in the original (a copy of object is copied in other object).
import utils   

#Using Object-oriented programming (OOP) paradigm to reduce the complexity of the system.
#By employing heavy re-usability of code and leveraging the concept of objects and classes.
#M must have exactly the same size (width, height) as T so we format the solution’s width and height.

class TetrisSolver:
    target = None
    shapeIds = None
    
    def __init__(self, target, shapeIds): #By using  “self” we can access the attributes and methods of the class.
        self.target = target
        self.shapeIds = shapeIds
    
    def Solve(self):
        width=len(self.target[0])
        height=len(self.target)
        tempMatrix = [[(0,0) for col in range(0, width)] for row in range(0, height)]
        
        targeti, initializedMatrix, pieceId = self.group(height, width, deepcopy(self.target), tempMatrix)
        
        maxpieceId = 0
        
        iterations = self.resolveSpan(width, height)

        for run in range (1, iterations):
            Mc, pieceIdm = self.selectShape(height, width,  deepcopy(targeti), deepcopy(initializedMatrix), deepcopy(pieceId))    

            if pieceIdm > maxpieceId:
                solution = Mc
                maxpieceId = pieceIdm    

        return solution    

#We have to resolve the average span of times iterations will occur.
    def resolveSpan(self, width, height):
        N2 = 0
        iterations = None      
        density = N2/(width*height)

        if width <= 100:   #Experimental running time: After testing the run rage according to each density and analyzing it.
            if density<= 0.3:
                iterations =1000 #the number of times it was run was modified to make it more efficient and increase its accuracy.
            if density<= 0.6:
                iterations =1000  
            if density<= 0.8:
                iterations =100     
        elif width <= 200:  #the bigger  the width the bigger the number of tiles.
            if density<= 0.3:
                iterations =100 #so for a better running time it is better to decrease the number of iterations.
            if density<= 0.6:
                iterations =50
            else:
                iterations =10
        elif width <= 500:
            iterations =10
        else:
            iterations =2 #for the maximum 1000x1000 2 iterations are more sufficient because the chance of getting a fitting piece is larger.
        return iterations

    def selectShape(self, height, width, targeta, matrix, pieceId):
        order=[]
        for row in range (0, height):
            order.append(row)
        for row in order: 
            for col in range(0, width): 
                if targeta[row][col]>1:
                    candidateShapeIds = self.identifyCandidateShapes(targeta, height, width, row, col)
                    
                    if candidateShapeIds !=[]:
                        selectedshapeID=candidateShapeIds[random.randint(0, len(candidateShapeIds)-1)]
                        targeta, matrix, pieceId = self.setPieceId(selectedshapeID, targeta, row, col, matrix, pieceId)
                            
        return matrix, pieceId

#The weight of a graph is the sum for all edges. 
#finding a group of tiles that have targets around it to fill them in.
    def group(self, height, width, CopyT, matrix):
        pieceId=1
        for row in range(0, height): 
            for col in range(0, width): 
                if CopyT[row][col]==1: 
                    if (self.isValidGridSize(CopyT, row, col, height, width)):
                        candidateShapeIds = self.identifyCandidateShapes(CopyT, height, width, row, col)
                        CopyT, matrix, pieceId = self.setPieceId(candidateShapeIds[0], CopyT, row, col, matrix, pieceId)
                    
        return CopyT, matrix, pieceId 
                                    
    def weightneighbour(self, queue, row, col, height, width, CopyT):
        if row < 0 or row > height or col < 0 or col > width:
            if CopyT[row][col] == 1:
                queue.append((row, col))
        return queue #collection of objects that supports FIFO

#Another weighting system was tried to be used to calculate the number of neighbors each tile has using kernel and 2D convolution.
    
#Inspired by the Kruskal and Dijkstra methods to find the shortest paths using breadth first search strategy.
#using a first-in, first-out (FIFO) strategy – a queue.
#The nodes discovered early are visited early.
#The graph is explored level by level, all levels (lecture).            
    def isValidGridSize(self, CopyT, row, col, height, width):
        queue = []
        queue.append((row, col))
        count = 0
        
        #checking if there are four cells available in the neighbour if yes then it can proceed (3 next to the main one).
        #BFS algorithm inspired by https://colorfulcodesblog.wordpress.com/2018/09/06/number-of-islands-tutorial-python/
        while queue:
            row, col = queue.pop() #inbuilt function that removes and returns last value from the list or the given index value
            CopyT[row][col] = 2
            count = count + 1            
            
            if row >= 1:  #to do row -1 we  should start with row >= 1
                queue=self.weightneighbour(queue, row - 1, col, height, width, CopyT)            
            if col >= 1: #to do col -1 we  should start with col >= 1
                queue=self.weightneighbour(queue, row, col - 1, height, width, CopyT)
            if row < height-1:
                queue=self.weightneighbour(queue, row + 1, col, height, width, CopyT)            
            if col < width-1:
                queue=self.weightneighbour(queue, row, col + 1, height, width, CopyT)            
        if count == 4:
            return True  #it can fit a piece.
        else:
            return False #it cannot fit a piece.
              
    def identifyCandidateShapes(self, CopyT, height, width, row, column):
        candidateShapes=[]
        for m in self.shapeIds: 
            valid = True 
            shape = utils.generate_shape(m) #import from utils the generate shape which is called by generate target.
            piece = [[y + row, x + column] for [y, x] in shape] 
            for [r, c] in piece: 
                if r < 0 or c < 0 or r >= height or c >= width:
                    valid = False
                    break #break terminates the current loop and resumes execution at the next statement
                if CopyT[r][c] == 0: #using boolean conditions.
                    valid = False
                    break
                
            if valid==True: 
                candidateShapes.append(m)                
        return candidateShapes
    
    def setPieceId(self, shapeId, CopyT, row, col, M, pieceId):
        shape=utils.generate_shape(shapeId)        
        for [y, x] in shape:
            CopyT[y + row][x + col] = 0
            M [y + row][x + col] = (shapeId, pieceId)
        pieceId+=1        
        return CopyT, M, pieceId 

def Tetris(target):  #removing forbidden shapes
    shapeIds =  [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    solver = TetrisSolver(target, shapeIds)
    
    return solver.Solve()



#This was the main code I tried to write while thinking out of the box but the running time for large matrices was slow so I decided to opt for a safer choice of algorithm architecture.

#import scipy.signal
#import numpy as np
#import utils
#placement_dtype = np.dtype([('shapeID', np.uint8), ('row', np.uint16), ('col', np.uint16), #(’score', np.uint8)])
#allowed_shapes = {}
#for i in range(4, 20):
    #allowed_shapes[i] = utils.generate_shape(i)
#class Problem():
    #current_piece_id = 1
    #def initialize_solution(self, target):
        #self.target = target
        #self.solution = [[None if col == 1 else (0,0) for col in row] for row in target]
    #def update_solution(self):
        #for i, row in enumerate(self.solution):
            #for j, col in enumerate(row):
                #if (col is None or col == (0, 0)) and self.partial_solution[i][j] is not None:
                    #self.solution[i][j] = self.partial_solution[i][j]
        #return []
    #def count_ones(self, target):
        #count = 0
        #for row in target:
            #for col in row:
                #if col == 1:
                    #count += 1
        #return count
    #def get_neighbours_matrix(self, target):
        #kernel = [[0,1,0],
                  #[1,0,1],
                  #0,1,0]]
        #neighbours_matrix = np.multiply(scipy.signal.convolve2d(target, kernel, #mode='same'), target)
        #return neighbours_matrix
    #def get_score_matrix(self, neighbours_matrix, shape):
        #neighbours_matrix[neighbours_matrix==0] = -50
        #neighbours_matrix[neighbours_matrix==1] = 500
        #neighbours_matrix[neighbours_matrix==2] = 150
        #neighbours_matrix[neighbours_matrix==3] = 50
        #neighbours_matrix[neighbours_matrix==4] = 0
        #score_matrix = scipy.signal.convolve2d(neighbours_matrix, np.flip(shape),mode='same')
        #return score_matrix
    #def generate_placements(self, neighbours_matrix, i, j):
        #placements = []
        #for shapeID, shape in allowed_shapes.items():
            #score_matrix = self.get_score_matrix(neighbours_matrix, shape)
            #score = score_matrix[i][j]
            #if(score>0):
                #placements.append((shapeID, i, j, score))
        #return placements
    #def apply_placement(self, placement, next_target):
        #placed = True
        #count = 0
        #for sq_row, sq_col in allowed_shapes[placement['shapeID']]:
            #row = placement['row'] + sq_row
            #col = placement['col'] + sq_col
            #try:
                #count += self.target[row][col]
           #except:
                #break
        #if count < 4:
            #return False
        #for sq_row, sq_col in allowed_shapes[placement['shapeID']]:
            #row = placement['row'] + sq_row
            #col = placement['col'] + sq_col
            #if(row>=len(self.partial_solution) or col>=len(self.partial_solution[0]) or row < 0 #or col < 0):
                #placed = False
                #break
            #if self.partial_solution[row][col]:
                #if self.partial_solution[row][col][0] == 0 and self.partial_solution[row][col] == 0:
                    #placed = False
                    #break
            #if self.solution[row][col]:
                #if self.solution[row][col][0] != 0:
                    #placed = False
                    #break
        #if placed:
            #for sq_row, sq_col in allowed_shapes[placement['shapeID']]:
                #row = placement['row'] + sq_row
                #col = placement['col'] + sq_col
                #self.partial_solution[row][col] = (placement['shapeID'], self.current_piece_id)
                #next_target[row][col] = 0
            #self.current_piece_id += 1
        #return placed
    #def update_target(self, next_target=None):
        #placed = False
        #if not next_target:
            #next_target = self.target.copy()
        #count_before = self.count_ones(next_target)
        #self.partial_solution = []
        #self.partial_solution = [[None if col == 1 else (0,0) for col in row] for row in #next_target]
        #neighbours_matrix = self.get_neighbours_matrix(next_target)
        #original_neighbours_matrix = neighbours_matrix.copy()
        #neighbours_matrix[neighbours_matrix!=1] = 0
        #for i in range(len(neighbours_matrix)):
            #for j in range(len(neighbours_matrix[i])):
                #placements = self.generate_placements(original_neighbours_matrix, i, j)
                #placements = np.array(placements, dtype=placement_dtype)
                #placements.sort(kind='mergesort', order='score')
                #placements = placements[::-1]
                #if placements.size != 0:
                    #for placement in placements:
                        #placed = self.apply_placement(placement, next_target)
                        #if placed:
                            #self.update_solution()
                            #break
                #if placed:
                    #break
            #if placed:
                #break
        #count_after = self.count_ones(next_target)
        #if placed or (count_after != count_before):
            #return self.update_target(next_target)
        #else:
            #return True
    #def solve(self):
        #self.update_target()
        #return True
#def Tetris(target):
    #problem = Problem()
    #problem.initialize_solution(target)
    #if(problem.solve()):
        #solution = [[el if el is not None else (0,0) for el in row] for row in problem.solution]
        #return solution

