import math
import random
from enum import Enum
import statistics
import pygame


#############################################################################################
# UTILITY FUNCTIONS

def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def mod2pi(angle):
    return angle % (2 * math.pi)

#############################################################################################
# MAIN OBJECTS

class Wall:
    def __init__(self, a: (float, float), b: (float, float)):
        self.a = a
        self.b = b

    def display(self, window):
        pygame.draw.line(window, (255, 255, 255), self.a, self.b)


class Car:
    def __init__(self, x: float, y: float, angle: float, velocity: float = 1):
        self.x = x
        self.y = y
        self.angle = angle
        self.velocity = velocity
        
        # Raycast depuis la voiture :
        self.rays = [
            Ray((self.x, self.y), mod2pi(self.angle + angle_change))
            for angle_change in (-math.pi / 2, -0.7, 0, 0.7, math.pi / 2)
        ]

    def turn(self, angle_change):
        self.angle = mod2pi(self.angle + angle_change)
        for ray in self.rays:
            ray.angle = mod2pi(ray.angle + angle_change)

    def update(self):
        self.x += math.cos(self.angle) * self.velocity
        self.y += math.sin(self.angle) * self.velocity
        for ray in self.rays:
            ray.x = self.x
            ray.y = self.y

    def display(self, window):
        a = 0.6
        l = 10
        points = [
            (self.x + math.cos(self.angle + angle_change)*l,
             self.y + math.sin(self.angle + angle_change)*l)
                for angle_change in [-a, a, math.pi-a, math.pi + a]]

        pygame.draw.polygon(window, (255, 255, 255), points, width = 2)
                

class Ray:
    """
        Used for raycasting and detecting walls
    """
    def __init__(self, coords: (float, float), angle: float):
        (self.x, self.y) = coords
        self.angle = angle

    def intersection(self, wall: Wall): # Mme Martineau serait fière x)
        x1 = wall.a[0]
        y1 = wall.a[1]
        x2 = wall.b[0]
        y2 = wall.b[1]
        x3 = self.x
        y3 = self.y
        x4 = self.x + math.cos(self.angle)
        y4 = self.y + math.sin(self.angle)

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0: return None # Le ray et le mur sont parallèles

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 * x3)) / den

        if 0 < t < 1: #and u > 0:
            i = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

            # Méthode schlag mais ça fonctionne donc bon..... :)
            d = dist(x3, y3, i[0], i[1])
            if  d > dist(x4, y4, i[0], i[1]):
                return i, d # retourne le point d'intersection et sa distance avec l'origine du ray

#############################################################################################
# ENVIRONEMENT

class Action(Enum):
    LEFT = 0
    FORWARD = 1
    RIGHT = 2

class Env:
    def initialize(self):
        self.car = Car(150, 150, random.random() * math.pi / 2)
        self.crash = False
        wall_coords = [
            ((100, 99), (100, 401)),
            ((99, 100), (701, 100)),
            ((99, 400), (701, 400)),
            ((700, 99), (700, 401)),

            ((199, 200), (601, 200)),
            ((200, 199), (200, 301)),
            ((600, 199), (600, 301)),
            ((199, 300), (601, 300)),
        ]
        self.walls = [Wall(c[0], c[1]) for c in wall_coords]
        return self.state()

    def state(self, window=None): # O(n²) -> Si j'ai le temps un jour il faudra améliorer ça...
        res = []
        for ray in self.car.rays:
            dist_min = math.inf
            inter = None

            for wall in self.walls:
                r = ray.intersection(wall) 
                if r is None : continue

                if r[1] < dist_min:
                    dist_min = r[1]
                    inter = r[0]

            res.append((inter, dist_min))

        if window is not None:
            for r in res:
                if r[0] is not None:
                    pygame.draw.line(window, (100, 100, 100), (self.car.x, self.car.y), (r[0][0], r[0][1]))

        return list(map(lambda x: x[1], res))
                
    def step(self, action: Action, window = None) -> int:
        """
            Retourne la récompense donné à l'agent sous la forme d'un entier
        """
        if self.crash: return -10, self.state(), self.crash

        if action == Action.LEFT:
            self.car.turn(-0.02)
        if action == Action.RIGHT:
            self.car.turn(0.02)

        self.car.update()

        res = self.state(window)
        for r in res:
            if r < 5: # Si la voiture touche un mur
                self.crash = True
                return -10, res, self.crash

        return min(res) / 20, res, self.crash

    def display(self, window):
        self.car.display(window)
        for wall in self.walls:
            wall.display(window)

#############################################################################################


def test_env(agent = None):
    pygame.init()
    background_color = (0, 0, 0)

    win = pygame.display.set_mode((800, 500))
    pygame.display.set_caption("Q-Car")
    
    env = Env()
    state = env.initialize()

    font = pygame.font.Font(None, 28)
    restart_text = font.render("space to restart", True, (150, 150, 150))
    arrow_text = font.render("arrow keys to steer", True, (150, 150, 150))

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    state = env.initialize()


        action = Action.FORWARD
        if agent is not None:
            a = int(agent.choose_action(agent.create_suitable_inputs([state])))
            if a == 0:
                action = Action.LEFT
            elif a == 1:
                action = Action.FORWARD
            elif a == 2:
                action = Action.RIGHT
            

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and keys[pygame.K_RIGHT]:
            action = Action.FORWARD
        elif keys[pygame.K_LEFT]:
            action = Action.LEFT
        elif keys[pygame.K_RIGHT]:
            action = Action.RIGHT

        win.fill(background_color)
        win.blit(restart_text, (10, 5))
        win.blit(arrow_text, (10, 25))

        reward, state, done = env.step(action, win)
        reward_text = font.render("reward : " + str(reward), True, (150, 150, 150))
        win.blit(reward_text, (10, 500 - 30))

        env.display(win)
        pygame.display.update()
        pygame.time.wait(10)

    pygame.quit()

if __name__ == "__main__":
    test_env()
