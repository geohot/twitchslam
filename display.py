import sdl2
import sdl2.ext

class Display(object):
  def __init__(self, W, H):
    sdl2.ext.init()

    self.W, self.H = W, H
    self.window = sdl2.ext.Window("twitch SLAM", size=(W,H)) #, position=(-500,-500))
    self.window.show()

  def paint(self, img):
    # junk
    events = sdl2.ext.get_events()
    for event in events:
      if event.type == sdl2.SDL_QUIT:
        exit(0)

    # draw
    surf = sdl2.ext.pixels3d(self.window.get_surface())
    surf[:, :, 0:3] = img.swapaxes(0,1)

    # blit
    self.window.refresh()

