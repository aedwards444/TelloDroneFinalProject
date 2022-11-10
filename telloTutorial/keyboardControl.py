from djitellopy import tello
import keyPressModule as kp
from time import sleep

kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())

def getKeyboardInput():
    # Variables for direction. 
    # lr = left and right
    # fb = forward and back
    # ud = up and down
    # yv = yaw velocity
    
    lr, fb, ud, yv = 0,0,0,0
    speed = 50
    
    if kp.getKey("LEFT"): 
        lr = -speed
    elif kp.getKey("RIGHT"): 
        lr = speed
    
    if kp.getKey("UP"):
        fb = speed
    elif kp.getKey("DOWN"):
        fb = -speed
    
    if kp.getKey("w"): 
        ud = speed
    elif kp.getKey("s"): 
        ud = -speed
        
    if kp.getKey("a"): 
        yv = speed
    elif kp.getKey("d"): 
        yv = -speed
        
    if kp.getKey("q"):
        me.land()
        
    if kp.getKey("e"):
        me.takeoff()
        
        
    return [lr, fb, ud, yv]

me.takeoff()

while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    sleep(0.05)
    