import sys
sys.path.insert(0, r'c:\Users\gavin\AppData\Local\RLBotGUIX\MyBots\Flamewall\src')
from bot_rl import FlamewallRL

bot = FlamewallRL('test', 0, 0)

class P: pass

p = P()
p.game_cars = [P() for _ in range(6)]

for car in p.game_cars:
    car.physics = P()
    car.physics.location = P()
    car.physics.location.x = car.physics.location.y = car.physics.location.z = 0
    car.physics.rotation = P()
    car.physics.rotation.pitch = car.physics.rotation.yaw = car.physics.rotation.roll = 0
    car.physics.velocity = P()
    car.physics.velocity.x = car.physics.velocity.y = car.physics.velocity.z = 0
    car.physics.angular_velocity = P()
    car.physics.angular_velocity.x = car.physics.angular_velocity.y = car.physics.angular_velocity.z = 0
    car.score_info = 0
    car.is_bot = True
    car.name = 'test'
    car.team = 0
    car.boost = 33
    car.hitbox = P()
    car.hitbox_offset = P()

p.game_cars[0].team = 0
for i, car in enumerate(p.game_cars):
    car.team = i % 2

p.game_ball = P()
p.game_ball.physics = P()
p.game_ball.physics.location = P()
p.game_ball.physics.location.x = p.game_ball.physics.location.y = p.game_ball.physics.location.z = 0
p.game_ball.physics.velocity = P()
p.game_ball.physics.velocity.x = p.game_ball.physics.velocity.y = p.game_ball.physics.velocity.z = 0
p.game_ball.physics.angular_velocity = P()
p.game_ball.physics.angular_velocity.x = p.game_ball.physics.angular_velocity.y = p.game_ball.physics.angular_velocity.z = 0

print('Testing get_output...')
result = bot.get_output(p)
print(f'Success! Got output: {result}')
