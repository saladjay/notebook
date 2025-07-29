import asyncio
import threading
import time
import traceback

class ServerProtocol(asyncio.Protocol):
    def __init__(self, server, **kwargs):
        super().__init__()
        self.server = server
        for k, v in kwargs.items():
            setattr(self, k, v)
        print(f"{self.__class__.__name__} Init ServerProtocol with kwargs: {kwargs}", flush=True)

    def connection_made(self, transport):
        """
        
        """
        self.transport = transport
        peername = transport.get_extra_info('peername')
        self.client_address = f"{peername[0]}:{peername[1]}"
        print(f"Connection from {self.client_address}", flush=True)
        self.reader = asyncio.StreamReader()
        self.writer = asyncio.StreamWriter(transport, self, self.reader, asyncio.get_event_loop())

        self.server.clients[self.client_address] = self
        print(f"Remaining clients: {len(self.server.clients)}", flush=True)
        asyncio.create_task(self.task())

    def data_received(self, data):
        self.reader.feed_data(data)

    def connection_lost(self, exc):
        print(f"Connection lost with {self.client_address}", flush=True)
        self.server.clients.pop(self.client_address, None)
        print(f"Remaining clients: {len(self.server.clients)}", flush=True)
        self.transport.close()

    async def task(self):
        while True:
            try:
                header = await asyncio.wait_for(self.reader.readexactly(4), timeout=30)
                data_length = int.from_bytes(await self.reader.readexactly(4), byteorder='big')
                data = await self.reader.readexactly(data_length)
                print(f"client_addr: {self.client_address} Header: {header} Received {data_length} bytes: {data}")
            except Exception as e:
                print(f"Error in handle_connection: {e}")
                traceback.print_exc()
            finally:
                self.transport.close()

class Server:
    def __init__(self, protocol_factory=ServerProtocol, host='127.0.0.1', port=11001, **kwargs):
        self.clients = {}
        self.host = host
        self.port = port
        self.protocol_factory = protocol_factory
        self.kwargs = kwargs
        self.server_thread = None
        for k, v in kwargs.items():
            setattr(self, k, v)
        print(f"{self.__class__.__name__} Init Server with kwargs: {kwargs}", flush=True)

    def create_protocol(self):
        return self.protocol_factory(self, **self.kwargs)
    
    async def run_server(self):
        loop = asyncio.get_running_loop()
        server = await loop.create_server(self.create_protocol, self.host, self.port)
        print(f'{self.__class__.__name__} Server created on {self.host}:{self.port}', flush=True) 
        async with server:
            await server.serve_forever()

    def run(self):
        asyncio.run(self.run_server())

    def run_on_thread(self):
        self.start_server()

    def start_server(self):
        # daemon=True means the thread will be terminated when the main thread is terminated
        self.server_thread = threading.Thread(target=self.run, daemon=True)
        self.server_thread.start()

if __name__ == '__main__':
    server = Server(port=11001)
    server.run_on_thread()
    while True:
        time.sleep(1)