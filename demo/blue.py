import asyncio
from bleak import BleakScanner, BleakClient
import threading

TARGET_DEVICE_NAME = "HMSoft"
HM_UART_SERVICE_UUID = "0000ffe0-0000-1000-8000-00805f9b34fb"
HM_TX_CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

class BlueModule():

    def __init__(self):
        self.target_device = None
        self.client = None

    async def find_dev(self):
        devices = await BleakScanner.discover()
        for device in devices:
            if device.name == TARGET_DEVICE_NAME:
                self.target_device = device
                break

        if not self.target_device:
            print("Target device not found.")
            return False
        return True

    async def connect_dev(self):
        if not self.target_device:
            if not await self.find_dev():
                return

        try:
            self.client = BleakClient(self.target_device.address)
            await self.client.connect()
            print("Bluetooth Connection Established.")
        except Exception as e:
            print(e)
            print("Retrying connection..")
            await asyncio.sleep(2)
            await self.connect_dev()

    async def disconnect_dev(self):
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            print("Bluetooth Connection Disconnected.")

    async def transmit(self, message):
        if self.client and self.client.is_connected:
            await self.client.write_gatt_char(HM_TX_CHARACTERISTIC_UUID, message.encode('ascii'))
        else:
            print("Device not connected.")

def get_connector():
    connector = BlueModule()
    return connector

def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(coro)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()

def send_message(connector, message):
    coro = connector.transmit(message)
    run_async(coro)

def disconnect_bluetooth(connector):
    coro = connector.disconnect_dev()
    run_async(coro)

# # Usage
# async def main():
#     blue_mod = BlueModule()
#     await blue_mod.connect_dev()
#     await blue_mod.transmit("Your message here")
#     await blue_mod.disconnect_dev()

# # Run the main function
# asyncio.run(main())
