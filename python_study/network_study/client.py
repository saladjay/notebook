import asyncio

class ClientProtocol(asyncio.Protocol):
    def __init__(self, client, **kwargs):
        self.client = client
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        print(f"{self.__class__.__name__} Init ClientProtocol with kwargs: {kwargs}", flush=True)
        self.closed =  asyncio.Future()

    def connection_made(self, transport):
        """
        
        """
        self.transport = transport
        self.server_address = transport.get_extra_info('peername')
        print('Connection made', self.server_address, flush=True)
        self.reader = asyncio.StreamReader()
        self.writer = asyncio.StreamWriter(transport, self, self.reader, asyncio.get_event_loop())
        asyncio.create_task(self.task())

    def data_received(self, data):
        self.reader.feed_data(data)

    def connection_lost(self, exc):
        print(f'Connection lost, {self.server_address}, Exception: "{exc}"', flush=True)
        self.transport.close()
        self.closed.set_result(True)

    async def task(self):
        while True:
            try:
                header = bytes([1, 2, 3, 4])
                data = b'hello world'
                self.writer.write(header + len(data).to_bytes(4, byteorder='big') + data)
                header  = await asyncio.wait_for(self.reader.readexactly(4), timeout=30)
                data_length = int.from_bytes(await self.reader.readexactly(4), 'big')
                data = await self.reader.readexactly(data_length)
                print(f"server_addr: {self.server_address} Header: {header} Received {data_length} bytes: {data}")
            except Exception as e:
                print(f"Error in handle_connection: {e}")
                continue

    async def wait_closed(self):
        try:
            await self.closed
        finally:
            print("client closed")

class Client:
    def __init__(self, protocol_factory=ClientProtocol, host='127.0.0.1', port=11234, auto_retry=True, **kwargs):
        self.host = host
        self.port = port
        self.protocol_factory = protocol_factory
        self.kwargs = kwargs
        self.auto_retry = auto_retry
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.kwargs['auto_retry'] = auto_retry

    def create_protocol(self):
        return self.protocol_factory(self, **self.kwargs)
    
    async def run(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                print(f"Connecting to {self.host}:{self.port}")
                transport, protocol = await loop.create_connection(self.create_protocol, self.host, self.port)
                print(f"Connected to {self.host}:{self.port}")
                await protocol.wait_closed()
            except Exception as e:
                if self.auto_retry:
                    print(f"Error in Client.run: {e}, retry after {self.timeout} seconds")
                    await asyncio.sleep(self.timeout)
                else:
                    raise e

if __name__ == '__main__':
    Client = Client(host='127.0.0.1', port=11001)
    asyncio.run(Client.run())