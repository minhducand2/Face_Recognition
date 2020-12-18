# import aiohttp, asyncio, json, sys

# async def send_name_async(session, name):
#   myobj = {'name': name}
#   data = json.dumps(myobj, ensure_ascii=True)
#   encoded_data = data.encode('utf-8')

#   async with session.post(url, data=encoded_data) as res:
#     print(res.status)
#     print(await res.text())

# async def main():
#   async with aiohttp.ClientSession() as session:
#     await send_name_async(session, name)
import json, requests, sys

def send_name(name):
    url = 'http://localhost:8080/txt2sp'
    myobj = {'name': name}

    data = json.dumps(myobj, ensure_ascii=True)

    encoded_data = data.encode('utf-8')
    
    r = requests.post(url, data=encoded_data,
                    headers={'Content-Type': 'application/json; charset=UTF-8'})

name = sys.argv[1]
url = 'http://localhost:8080/txt2sp'

send_name(name)

# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())

# asyncio.run(main())