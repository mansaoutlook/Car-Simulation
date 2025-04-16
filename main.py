import pygame    #python library to handle 2d game
import numpy as np  
import time
import math
import pickle   #to save/load trained nueral network
import random   #mutation
import copy     #to clone same brain as best one
from utils import scale_image, blit_rotate_center

pygame.font.init()  
pygame.init()           #used for input,sound,grsphic in pygame

GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)  #image to 2.5 time badei krdi
TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)  #image ko 0.9 time kardiya size main

TRACK_BORDER = scale_image(pygame.image.load('imgs/track-border.png'),0.9)  #same as track size

SENSOR_COLOR = (255, 255, 255,0)  #rbg colur ko boundary manke collision check karega, white ki sensitivity 0 hain

FINISH = pygame.image.load("imgs/finish.png")   
FINISH_MASK = pygame.mask.from_surface(FINISH)   
FINISH_POSITION = (130, 250)                     #finsing line daldi

RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.4)       
GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.4)

WIDTH, HIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HIGHT)) #Displaykrdiya track ki height aur width ke hisab se
pygame.display.set_caption("Racing Game!")    

MAIN_FONT = pygame.font.SysFont('comicsans', 44)       
FPS=60          #frames per second 

RED_SURF = RED_CAR.convert_alpha()  
RED_RECT = RED_SURF.get_rect(center = (0,0))
RED_MASK = pygame.mask.from_surface(RED_CAR)

TRACK_BORDER_SURF = TRACK_BORDER.convert_alpha()
TRACK_BORDER_POS = (0,0)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

COINS = np.array(((172, 109),(111, 64),(57, 107),(57, 464),(306, 702),
                 (398, 666),(427, 493),(509, 462),(583, 518),(608, 692),
                 (728, 675),(723, 375),(418, 347),(426, 250),(716, 232),
                 (722, 78),(289, 73),(257, 379),(173, 362)))

class AbstractCar:
    def _init_(self, max_vel, rotation_vel):
        self.max_vel = max_vel            
        self.rotation_vel = rotation_vel        #velocity of roation
        self.img = self.IMG                    
        self.x, self.y = self.START_POS         
        self.vel = 0
        self.angle = 0              
        self.acceleration = 0.1             
    def rotate(self, left=False, right=False): 
        if left==True:
            self.angle+= self.rotation_vel    #clockwise
        if right==True: 
            self.angle-= self.rotation_vel   #anticlock
    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)  #current pos se rotate krke rotate hone ke badki image bana di 
    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)       #max velocity
        self.move()               
    def move_backward(self):                                               
        self.vel = max(self.vel - self.acceleration, -self.max_vel/2)   #piche gya toh
        self.move() 
    def move(self):
        rad = math.radians(self.angle)                                 
        vertical = math.cos(rad) * self.vel
        horizontal = math.sin(rad) * self.vel    
        self.y -= vertical
        self.x -= horizontal
    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration/2, 0)   
        self.move()
    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)    #grid of pixel main convert krdiya ,like tranp=0,opaque=1; toh collision pata chale
        offset = (int(self.x-x), int(self.y-y))          #game pe actual pos of car
        poi = mask.overlap(car_mask, offset)              #collision check
        return poi                                  
    def reset(self):                                     
        self.x, self.y = self.START_POS
        self.vel = 0
        self.angle = 0
class PlayerCar(AbstractCar):
    IMG = RED_SURF
    START_POS = (180, 200)
    def _init_(self,max_vel, rotation_vel):
        super()._init_(max_vel, rotation_vel)
        self.f_x = int(self.x + self.img.get_width()/2)   #cenntres of car 
        self.f_y = int(self.y + self.img.get_height()/2)
        self.s_lenght = 0           #length of ray
        self.s_count = 5           #5 sensor
        self.dis=np.zeros((self.s_count))     #harek sensor ke liye data store karega
        self.brain = NeuronalNetwork([self.s_count,6,4])  #5 sensors/ 6 hidden neuron/4 hidden neurons
        self.damage = False
        self.max_dis = 100    #485 max
        self.points = 0       #point kitne milne
        self.points_pos = 0   #agla point kha hain
        self.time_s = time.perf_counter()  
    def bounce(self):
        self.vel = -self.vel     
        self.move()
    def ai_collition(self):
        self.img = GREEN_CAR   #obstacle pe hit
        self.damage = True     
    def sensor(self,win):            #5 sensors ki diatnce wall se
        s_x=int(self.x + self.img.get_width()/2)    #sensors centre of car se nikal rhe
        s_y=int(self.y + self.img.get_height()/2)
        for i in range(self.s_count):
            if self.s_count!=1:
                theta = math.radians(self.angle)+i*(math.pi/(self.s_count-1))  #ans+pi/3,2*pi/3,pi
            else:
                theta = math.radians(self.angle)+(math.pi/2)    #anshle+pi/2
            self.s_lenght=0                                      #current length of sensor
            self.f_x = int(s_x+self.s_lenght*math.cos(theta))   #tip of sensor
            self.f_y = int(s_y-self.s_lenght*math.sin(theta))   
            while TRACK_COPY.get_at((self.f_x,self.f_y)) == SENSOR_COLOR:  #tip ka colur same hain toh sensor mtlb wall tak
                self.f_x = int(s_x+self.s_lenght*math.cos(theta))
                self.f_y = int(s_y-self.s_lenght*math.sin(theta))        
                self.s_lenght +=1                            #usi direction main length badhate jao
            if self.s_lenght > self.max_dis:             
                self.s_lenght = self.max_dis
                self.f_x = int(s_x+self.s_lenght*math.cos(theta))
                self.f_y = int(s_y-self.s_lenght*math.sin(theta))     
            pygame.draw.line(win,(0,255,0),
                        (s_x, s_y),             #sensor origin
                        (self.f_x,self.f_y), 3)  #sensor tip and thickness
            self.dis[i]=self.s_lenght/self.max_dis   #max_sensor_dis_travel/max_allowed_to_trav  1 no obsta,0 -> colliderange[0,1]
    def get_dist(self):      
        return self.dis    #return all sensor dis[0,1]*mx_allo
    def train_brain(self):  #current sensor dis dala
        return self.brain.feedForward(self.dis)
    def get_points(self):             
        point_dist = math.sqrt(
            (self.x-COINS[self.points_pos][0])*2+(self.y-COINS[self.points_pos][1])*2) #coind pos se distance nikala
        if point_dist < 50:                   
            self.points+=1       
            self.points_pos+=1       #next indes of coin pos
            self.time_s = time.perf_counter()  #timer phir se start
            if self.points_pos == len(COINS):   #sare collect karliye toh finsih tak pahuch gya
                self.points_pos = 0             
    def time_out(self):        
        now = time.perf_counter()
        if now-self.time_s > 5:    #5 sec se jyada ho gya toh car damaged hain
            self.damage =True
            self.img = GREEN_CAR
    def update_brain(self,new_b):
        self.brain = copy.deepcopy(new_b)    #best car copied
    def restar(self):          
        self.img = RED_CAR
        self.x, self.y = self.START_POS
        self.vel = 0
        self.angle = 0
        self.damage = False
        self.points = 0
        self.points_pos = 0
        self.time_s = time.perf_counter()
class Layer:
    def _init_(self,inputCount, outputCount):
        self.inputs = np.zeros(inputCount)                  #input
        self.outputs = np.zeros(outputCount)                #output     
        self.biases = np.random.rand(outputCount)*2-1                #random weight and bias allocation [-1,1]
        self.weights = np.random.rand(inputCount,outputCount)*2-1
    def feedForward(self, givenInputs):
        self.inputs = np.array(givenInputs)  
        for i in range(self.outputs.shape[0]):
            sum = 0         
            for j in range(self.inputs.shape[0]):
                #print('i=',i,'j=',j)
                sum+=self.inputs[j]*self.weights[j,i]  #summation wi*xi
            if sum > self.biases[i]:                    #threshold base binary function
                self.outputs[i] = 1
            else:
                self.outputs[i] = 0
        return self.outputs
class NeuronalNetwork:
    def _init_(self, neuronCounts):
        self.layers = []
        for i in range(len(neuronCounts)-1):
            self.layers.append(Layer(neuronCounts[i], neuronCounts[i+1])) #[i,h,o] i-h aur h-o ke bich layer
    def feedForward(self,givenInputs): 
        outputs = self.layers[0].feedForward(givenInputs)          
       
        for i in range(1,len(self.layers)):
            outputs = self.layers[i].feedForward(outputs)   # feedforward chala diya yeha pe            
        return outputs
    def mutate(self, amount=0.5):
        for layer in self.layers:
            for i in range(layer.biases.shape[0]):
                layer.biases[i]=np.interp(amount,
                                          [-1,1],
                                          [layer.biases[i],random.random()*2-1])    #old,new kitna pass hain dono usse mutate kiya 0.5 tak 
            for i in range(layer.weights.shape[0]):
                for j in range(layer.weights.shape[1]):
                    layer.weights[i][j]=np.interp(amount,
                                                  [-1,1],
                                                  [layer.weights[i][j],random.random()*2-1])



def draw(win, images, player_car):
    for img, pos in images:
        win.blit(img, pos)  #harek image ko uski pos aur img pe draw kiya
    player_car.draw(win)     
    player_car.sensor(win) #SENSOR 
    for i in COINS:
        pygame.draw.circle(win,(255,0,0),i,5)  #coins dal diye sari location pe
    pygame.display.update()

def draw_ai(win, images, AIs):
    for img, pos in images:         
        win.blit(img, pos)    #backgroung image
    for car in AIs:
        car.draw(win)                 
        if car.damage == False:
            car.sensor(win)
            car.get_points()  #coins leliya close hain toh
            car.time_out()    #5 sec se kuch nhi liya toh damage true
            output = car.train_brain()     #sensor daldiya 
            move_ai(output,car)    
        found = handle_collision(car,controlType='AI')
    damaged = [car.damage for car in AIs]                
    if all(damaged) or found==True:            #sare cars damaged hain ya finish line pahuch gye toh epo terminate
        print('Epoca Termino!')
        best_car = max(AIs,key=lambda x:x.points)           #max points collcet karne wala best car  
        print('Best=',best_car,' Points=',best_car.points)        
        for i in range(len(AIs)):                             
            AIs[i].update_brain(best_car.brain)            
            if i != 0:
                AIs[i].brain.mutate(amount=0.1)           #ek car chodke sabko mutate kardiya
            AIs[i].restar()                              #start pos pe daldiya
    pygame.display.update()                               #drawing change kiya sab start pe pahuch gye

def move_ai(output,player_car):
    moved = False
    if output[0] == 1:                      
        player_car.rotate(left= True)
    if output[1] == 1:
        player_car.rotate(right=True)
    if output[2] == 1:
        moved = True
        player_car.move_forward()
    if output[3] == 1:
        moved = True
        player_car.move_backward()
    if not moved:
        player_car.reduce_speed()

def handle_collision(player_car,controlType):
    if player_car.collide(TRACK_BORDER_MASK) != None:
        if controlType=='AI':
            player_car.ai_collition()
        else:
            player_car.bounce()
        pass

    finish_poi_collide = player_car.collide(FINISH_MASK, *FINISH_POSITION)
    if finish_poi_collide != None:
        if finish_poi_collide[1] == 0:
            player_car.bounce()
        else:
            print('Best=',player_car,' Points=',player_car.points)
            save_car(player_car.brain)
            print("Finish")
            return True
    return False

def create_rewards(player_car,coins):
    keys = pygame.key.get_pressed()
    press = pygame.mouse.get_pressed()
    point = pygame.mouse.get_pos()
    if press[0]==True:
        print('P=',(int(player_car.x), int(player_car.y)))

def save_car(obj):
    try:
        with open("data_bestone.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)

run=True
clock = pygame.time.Clock()
images=[(GRASS, (0,0)), 
        (FINISH, FINISH_POSITION), 
        (TRACK_BORDER, TRACK_BORDER_POS)]
player_car = PlayerCar(max_vel=4, rotation_vel=4)
TRACK_COPY = TRACK_BORDER.convert_alpha()
#Here handle events
controlType='AI'
AIs_num = 100
if controlType == 'AI':
    AIs=[]
    for i in range(AIs_num):
        AIs.append(PlayerCar(max_vel=4, rotation_vel=4))
max_value = 0
tic = time.perf_counter()
#obj = load_object("data.pickle")
while True:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print('Cerrando')
            run = False
            pygame.quit()
            exit()

    WIN.fill('black')
    toc = time.perf_counter()
    if controlType == "AI":
        draw_ai(WIN, images, AIs)
    else:
        draw(WIN, images, player_car)
       
