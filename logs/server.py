import asyncio
import websockets

def write(message: str, filename: str):
    f = open(filename, "a")
    f.write(message + "\n")
    f.close()

async def hello(websocket, path):
    message = await websocket.recv()
    print(message)
    if "|" in message:
        entry = message[14:20] + "," + message[41:]
        write(message=entry, filename="metrics.csv")
    write(message=message, filename="system.log")

start_server = websockets.serve(hello, "0.0.0.0", 7777)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()