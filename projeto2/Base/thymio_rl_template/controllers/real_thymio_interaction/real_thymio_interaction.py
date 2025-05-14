try:
    import time
    import threading
    import gymnasium as gym
    import numpy as np
    from tdmclient import ClientAsync

except ImportError:
    sys.exit('Please make sure you have all dependencies installed.')


left_motor = 0.0
right_motor = 0.0
prox_sensor = None
ground_sensor = None

def run_thymio_program():
    global prox_sensor
    with ClientAsync() as client:
        async def prog():
            global prox_sensor
            global ground_sensor
            with await client.lock() as node:
                await node.wait_for_variables({"prox.horizontal","prox.ground.reflected"})
                while True:
                    prox_sensor = node.v.prox.horizontal
                    ground_sensor = node.v.prox.ground.reflected
                    node.v.motor.left.target = int(left_motor)
                    node.v.motor.right.target = int(right_motor)
                    node.flush()
                    await client.sleep(0.1)
        client.run_async_program(prog)


def main():
    global left_motor;
    global right_motor;

    thymio_thread = threading.Thread(target=run_thymio_program)
    
    thymio_thread.start()

    i = 0
    
    left_motor = right_motor = 1.0

    while(True):
        
        if prox_sensor is not None and ground_sensor is not None:
        
            right_motor = ground_sensor[1]/10 - prox_sensor[0]/10
            left_motor = ground_sensor[0]/10 - prox_sensor[4]/10
            
            print("Sensors: ", prox_sensor[0], ground_sensor[0], " Actuators: ", left_motor, right_motor)
            
            i = i + 1

            time.sleep(0.01)
        

if __name__ == '__main__':
    main()
