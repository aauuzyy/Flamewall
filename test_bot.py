import sys
sys.path.insert(0, 'src')

# Force reload to get fresh code
import importlib
if 'bot_rl' in sys.modules:
    importlib.reload(sys.modules['bot_rl'])

from bot_rl import FlamewallRL
from rlbot.utils.structures.game_data_struct import GameTickPacket

print("Initializing bot...")
bot = FlamewallRL('Test', 0, 0)
bot.initialize_agent()

print("\nTesting get_output with empty packet...")
packet = GameTickPacket()
try:
    output = bot.get_output(packet)
    print("✓ get_output() works!")
    print(f"Output: throttle={output.throttle}, steer={output.steer}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
