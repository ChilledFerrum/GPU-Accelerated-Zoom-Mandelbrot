import pygame
import numpy as np

import cv2 as cv
import taichi as ti

width, height = 1920, 1080
offset = np.array([1.3 * width, height]) // 2
zoom = 2.2 / height
output_folder = "out/"

texture = "textures/Texture1.jpg"
textures = pygame.image.load(texture)  # Best visualized texture texture1.jpg

texture_size = min(textures.get_size()) - 1
texture_array = pygame.surfarray.array3d(textures)

@ti.data_oriented
class Mandelbrot:
    def __init__(self, App, _surface):
        self.app = App
        self.screen = None
        self.surface = _surface
        self.max_iter, self.max_iter_limit = 500, 5500

        ti.init(arch=ti.gpu)
        self.screen_field = ti.Vector.field(3, ti.uint8, (width, height))
        self.texture_field = ti.Vector.field(3, ti.uint8, textures.get_size())
        self.texture_field.from_numpy(texture_array)

        self.velocity = 0.015
        self.zoom, self.scale = 2.5/height, 0.993
        self.increment = ti.Vector([0.0, 0.0])

        self.x = np.linspace(0, width, num=width, dtype=np.float32)
        self.y = np.linspace(0, height, num=height, dtype=np.float32)
        self.width = width
        self.height = height


        self.app_speed = 1 / 4000
        self.prev_time = pygame.time.get_ticks()

    @ti.kernel
    def render(self, max_iter: ti.int32, zoom: ti.float32, dx: ti.float32, dy: ti.float32):
        for x, y in self.screen_field:
                c = ti.Vector([(x - offset[0]) * zoom - dx, (y - offset[1]) * zoom - dy])
                z = ti.Vector([0.0, 0.0])
                num_iter = 0
                for i in range(max_iter):
                    z = ti.Vector([(z.x ** 2 - z.y ** 2 + c.x), (2 * z.x * z.y + c.y)])
                    if z.dot(z) > 4:
                        break
                    num_iter += 1
                col = int(texture_size * num_iter / max_iter)
                self.screen_field[x, y] = self.texture_field[col, col]

    def delta_time(self):
        time_now = pygame.time.get_ticks() - self.prev_time
        self.prev_time = time_now
        return time_now * self.app_speed

    def inputs(self):
        key = pygame.key.get_pressed()
        dt = self.delta_time()
        if key[pygame.K_a]:
            self.increment[0] += self.velocity * dt
            print("X=", self.increment[0])
        if key[pygame.K_d]:
            self.increment[0] -= self.velocity * dt
            print("X=", self.increment[0])
        if key[pygame.K_w]:
            self.increment[1] += self.velocity * dt
            print("Y=", self.increment[1])
        if key[pygame.K_s]:
            self.increment[1] -= self.velocity * dt
            print("Y=", self.increment[1])

        if key[pygame.K_q] or key[pygame.K_e]:
            if key[pygame.K_q]:
                self.zoom *= self.scale
                self.velocity *= self.scale
            if key[pygame.K_e]:
                inv_scale = 2 - self.scale
                self.zoom *= inv_scale

        if key[pygame.K_LEFT]:
            self.max_iter -= 10
        if key[pygame.K_RIGHT]:
            self.max_iter += 10
        self.max_iter = min(max(self.max_iter, 2), self.max_iter_limit)

        if key[pygame.K_SPACE]:
            img = pygame.Surface((width, height))
            img.blit(self.surface, (0, 0), ((0, 0), (width, height)))

            pygame.image.save(img, output_folder+"Result.png")


    def update(self):
        self.inputs()

        self.render(self.max_iter, self.zoom, self.increment[0], self.increment[1])
        self.screen = self.screen_field.to_numpy()

    def draw(self):
        pygame.surfarray.blit_array(self.app.screen, self.screen)

    def generateMandelbrot(self):
        self.update()
        self.draw()


class App:
    def __init__(self):
        # self.fps = fps

        # Pygame Initializations
        pygame.init()
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.K_ESCAPE, pygame.K_SPACE])
        self.clock = pygame.time.Clock()

        # Set Canvas Size & Title
        self.CanvasSize = width, height
        self.screen = pygame.display.set_mode(self.CanvasSize, pygame.SCALED)

        self.mandelbrot = Mandelbrot(self, self.screen)
        self.running = True

    def run(self):
        while self.running:
            pygame.display.flip()
            self.mandelbrot.generateMandelbrot()

            [exit() for i in pygame.event.get() if i.type == pygame.QUIT]
            self.clock.tick()
            pygame.display.set_caption(f'FPS: {self.clock.get_fps() :.2f}')


if __name__ == "__main__":
    canvas = App()
    canvas.run()
